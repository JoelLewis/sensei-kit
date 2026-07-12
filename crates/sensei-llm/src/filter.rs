//! Incremental filtering of Gemma 4 channel blocks from streamed output.
//!
//! Gemma 4 models can emit auxiliary channels (e.g. `<|channel>thought ...
//! <channel|>`) before the visible answer. Thinking is opt-in via the chat
//! template and [`crate::chat::format_chat`] never requests it, but the
//! filter guards against a model emitting a channel block anyway so it never
//! reaches the UI.

const CHANNEL_START: &str = "<|channel>";
const CHANNEL_END: &str = "<channel|>";

/// Strips `<|channel>...<channel|>` blocks from text that arrives in pieces.
///
/// Marker sequences may be split across pieces, so a small tail is buffered
/// until it can be classified as visible text or a channel marker.
#[derive(Debug, Default)]
pub struct ChannelFilter {
    pending: String,
    in_channel: bool,
}

impl ChannelFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a new piece of generated text; returns the visible portion.
    pub fn push(&mut self, piece: &str) -> String {
        self.pending.push_str(piece);
        let mut visible = String::new();

        loop {
            if self.in_channel {
                if let Some(pos) = self.pending.find(CHANNEL_END) {
                    self.pending.drain(..pos + CHANNEL_END.len());
                    self.in_channel = false;
                } else {
                    // Discard channel content, keeping only a possible partial end marker.
                    let keep = partial_marker_suffix(&self.pending, CHANNEL_END);
                    self.pending.drain(..self.pending.len() - keep);
                    return visible;
                }
            } else if let Some(pos) = self.pending.find(CHANNEL_START) {
                visible.push_str(&self.pending[..pos]);
                self.pending.drain(..pos + CHANNEL_START.len());
                self.in_channel = true;
            } else {
                // Emit everything except a possible partial start marker.
                let keep = partial_marker_suffix(&self.pending, CHANNEL_START);
                let emit_to = self.pending.len() - keep;
                visible.push_str(&self.pending[..emit_to]);
                self.pending.drain(..emit_to);
                return visible;
            }
        }
    }

    /// Flush any buffered text at end of generation.
    ///
    /// An unterminated channel block is dropped entirely.
    pub fn finish(mut self) -> String {
        if self.in_channel {
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }
}

/// Length of the longest strict marker prefix that `text` ends with.
///
/// Markers are ASCII, so the returned length is always at a char boundary.
fn partial_marker_suffix(text: &str, marker: &str) -> usize {
    let max = (marker.len() - 1).min(text.len());
    (1..=max)
        .rev()
        .find(|&n| text.ends_with(&marker[..n]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[&str]) -> (String, String) {
        let mut filter = ChannelFilter::new();
        let mut streamed = String::new();
        for piece in pieces {
            streamed.push_str(&filter.push(piece));
        }
        let tail = filter.finish();
        (streamed.clone(), format!("{streamed}{tail}"))
    }

    #[test]
    fn plain_text_passes_through() {
        let (_, full) = run(&["Control the ", "center with e4."]);
        assert_eq!(full, "Control the center with e4.");
    }

    #[test]
    fn strips_channel_block() {
        let (_, full) = run(&["<|channel>thought\nsome hidden reasoning<channel|>The answer."]);
        assert_eq!(full, "The answer.");
    }

    #[test]
    fn strips_channel_block_split_across_pieces() {
        let (_, full) = run(&[
            "<|chan",
            "nel>thought\nhidden",
            " reasoning<chan",
            "nel|>Visible text.",
        ]);
        assert_eq!(full, "Visible text.");
    }

    #[test]
    fn unterminated_channel_is_dropped() {
        let (_, full) = run(&["Before. <|channel>thought\nnever closed"]);
        assert_eq!(full, "Before. ");
    }

    #[test]
    fn partial_marker_lookalike_is_emitted() {
        // A '<' that never becomes a marker must still be emitted.
        let (_, full) = run(&["a < b and a <| weird token"]);
        assert_eq!(full, "a < b and a <| weird token");
    }

    #[test]
    fn grammar_style_tags_pass_through() {
        // GBNF-constrained output uses tags like <classification> and
        // <coaching>; they must never be mistaken for channel markers.
        let (_, full) = run(&[
            "<classification>{\"error_class\": \"Direction\"}</classification>\n",
            "<coach",
            "ing>Watch the whole board.</coaching>",
        ]);
        assert_eq!(
            full,
            "<classification>{\"error_class\": \"Direction\"}</classification>\n\
             <coaching>Watch the whole board.</coaching>"
        );
    }

    #[test]
    fn text_around_multiple_blocks() {
        let (_, full) = run(&["One <|channel>x<channel|>two <|channel>y<channel|>three"]);
        assert_eq!(full, "One two three");
    }

    #[test]
    fn streaming_never_leaks_channel_content() {
        let mut filter = ChannelFilter::new();
        let mut streamed = String::new();
        for piece in ["<|channel>th", "ought\nsecret<channel|>", "ok"] {
            streamed.push_str(&filter.push(piece));
        }
        assert!(!streamed.contains("secret"));
        assert_eq!(format!("{streamed}{}", filter.finish()), "ok");
    }

    #[test]
    fn partial_marker_suffix_lengths() {
        assert_eq!(partial_marker_suffix("abc<|chan", CHANNEL_START), 6);
        assert_eq!(partial_marker_suffix("abc<", CHANNEL_START), 1);
        assert_eq!(partial_marker_suffix("abc", CHANNEL_START), 0);
        assert_eq!(partial_marker_suffix("", CHANNEL_START), 0);
    }
}
