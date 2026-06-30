from app.config import AppSettings
from app.text_processor import TextProcessor


def _processor(chunk_size: int = 10, overlap: int = 2) -> TextProcessor:
    return TextProcessor(
        AppSettings(chunk_size=chunk_size, chunk_overlap=overlap, _env_file=None)
    )


def test_html_to_text_extracts_block_tags_in_order():
    html = """
    <article>
      <h1>Title</h1>
      <p>First &amp; only paragraph.</p>
      <ul><li>One</li><li>Two</li></ul>
      <blockquote>Quoted.</blockquote>
      <div>Ignored div text.</div>
    </article>
    """
    text = _processor().html_to_text(html)
    parts = text.split("\n\n")
    assert parts == ["Title", "First & only paragraph.", "One", "Two", "Quoted."]


def test_html_to_text_collapses_whitespace():
    text = _processor().html_to_text("<p>  hello\n\n   world\t\t!  </p>")
    assert text == "hello world !"


def test_html_to_text_empty_input():
    p = _processor()
    assert p.html_to_text("") == ""
    assert p.html_to_text("   ") == ""
    assert p.html_to_text("<div></div>") == ""


def test_chunk_text_respects_paragraph_boundaries_with_overlap():
    text = "alpha beta gamma delta\n\nepsilon zeta eta theta\n\niota kappa lambda mu"
    chunks = _processor(chunk_size=6, overlap=2).chunk_text(text)
    # First chunk fills with paragraph 1 (4 words); adding paragraph 2 (4 words)
    # would push to 8 words > 6, so chunk 1 emits with 4 words, overlap=2 from end.
    assert chunks[0] == "alpha beta gamma delta"
    assert chunks[1].startswith("gamma delta epsilon zeta eta theta")


def test_chunk_text_no_overlap_when_overlap_zero():
    text = "one two three four\n\nfive six seven eight"
    chunks = _processor(chunk_size=4, overlap=0).chunk_text(text)
    assert chunks == ["one two three four", "one two three four five six seven eight"]


def test_chunk_text_empty_returns_empty():
    assert _processor().chunk_text("") == []
    assert _processor().chunk_text("   \n\n   ") == []


def test_chunk_text_single_short_paragraph():
    assert _processor(chunk_size=100).chunk_text("hello world") == ["hello world"]
