import os
import tempfile
from src.scraper import extract_page_content, extract_pdf_text, discover_links


def test_extract_page_content_returns_title_and_text():
    html = """
    <html>
    <head><title>Form 1040 Instructions</title></head>
    <body>
        <nav>Skip this nav</nav>
        <div id="main-content">
            <h1>Form 1040 Instructions</h1>
            <p>Use Form 1040 to file your individual tax return.</p>
        </div>
        <footer>Skip this footer</footer>
    </body>
    </html>
    """
    result = extract_page_content(html, "https://www.irs.gov/forms-instructions/form-1040")
    assert result["title"] == "Form 1040 Instructions"
    assert "Form 1040" in result["content"]
    assert "Skip this nav" not in result["content"]
    assert "Skip this footer" not in result["content"]
    assert result["url"] == "https://www.irs.gov/forms-instructions/form-1040"


def test_extract_page_content_fallback_to_body():
    html = """
    <html>
    <head><title>Some Page</title></head>
    <body>
        <p>Main content here.</p>
    </body>
    </html>
    """
    result = extract_page_content(html, "https://www.irs.gov/some-page")
    assert "Main content here" in result["content"]


def test_extract_pdf_text(tmp_path):
    bad_file = tmp_path / "not_a_pdf.pdf"
    bad_file.write_text("this is not a pdf")
    result = extract_pdf_text(str(bad_file))
    assert result == ""


def test_discover_links_finds_irs_links():
    html = """
    <html><body>
        <a href="/forms-pubs/about-form-1040">Form 1040</a>
        <a href="/forms-pubs/about-form-w-2">Form W-2</a>
        <a href="https://external.com/page">External</a>
    </body></html>
    """
    links = discover_links(html, "https://www.irs.gov/forms-instructions")
    assert "https://www.irs.gov/forms-pubs/about-form-1040" in links
    assert "https://www.irs.gov/forms-pubs/about-form-w-2" in links
    assert "https://external.com/page" not in links
