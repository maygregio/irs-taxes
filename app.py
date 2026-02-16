# app.py
import os
import hmac
import streamlit as st
from src.chain import ask
from src.scraper import scrape_irs
from src.indexer import build_index

st.set_page_config(page_title="IRS Tax Assistant", page_icon="📋", layout="centered")


def check_password():
    """Prompt for a password and return True if correct."""
    if st.session_state.get("authenticated"):
        return True

    password = st.text_input("Password", type="password", placeholder="Enter password to continue")
    if password:
        correct = os.environ.get("APP_PASSWORD", "")
        if not correct:
            st.error("APP_PASSWORD is not configured.")
            return False
        if hmac.compare_digest(password, correct):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

# Material UI inspired styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* ── Global / Material foundations ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp {
        background: #f5f5f5;
    }

    /* ── Tame all heading sizes ── */
    h1 { font-size: 1.25rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    h4, h5, h6 { font-size: 0.88rem !important; }
    .chat-bubble h1 { font-size: 1.05rem !important; }
    .chat-bubble h2 { font-size: 0.95rem !important; }
    .chat-bubble h3 { font-size: 0.88rem !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }

    /* ── App bar / Header ── */
    .mui-appbar {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        color: white;
        padding: 0.85rem 1.5rem;
        margin: -1rem -1rem 1.25rem -1rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 2px 8px rgba(21, 101, 192, 0.3);
        text-align: center;
    }
    .mui-appbar h1 {
        font-size: 1.15rem !important;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
        color: white;
    }
    .mui-appbar p {
        font-size: 0.8rem;
        opacity: 0.85;
        margin: 0.25rem 0 0 0;
        font-weight: 300;
        color: white;
    }

    /* ── Custom chat bubbles (HTML rendered history) ── */
    .chat-row {
        display: flex;
        margin-bottom: 1rem;
        align-items: flex-end;
        gap: 6px;
    }
    .chat-row.user {
        justify-content: flex-end;
    }
    .chat-row.assistant {
        justify-content: flex-start;
    }
    .chat-avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    .chat-avatar.assistant-av {
        background: #e3f2fd;
        color: #1565c0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .chat-avatar.user-av {
        background: #1976d2;
        color: white;
        box-shadow: 0 1px 3px rgba(25,118,210,0.3);
    }
    .chat-bubble {
        max-width: 75%;
        padding: 0.85rem 1.1rem;
        font-size: 0.92rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    .chat-bubble.assistant-bubble {
        background: #ffffff;
        color: #212121;
        border-radius: 18px 18px 18px 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .chat-bubble.user-bubble {
        background: linear-gradient(135deg, #1976d2, #1565c0);
        color: #ffffff;
        border-radius: 18px 18px 4px 18px;
        box-shadow: 0 2px 8px rgba(25,118,210,0.3);
    }
    .chat-bubble p { margin: 0 0 0.5rem 0; }
    .chat-bubble p:last-child { margin-bottom: 0; }
    .chat-bubble ul, .chat-bubble ol { margin: 0.25rem 0; padding-left: 1.25rem; }
    .chat-bubble li { margin-bottom: 0.2rem; }
    .chat-bubble strong { font-weight: 600; }
    .chat-bubble em { font-style: italic; }
    .chat-bubble code {
        background: rgba(0,0,0,0.06);
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .user-bubble code {
        background: rgba(255,255,255,0.18);
    }
    .chat-bubble a { color: #1565c0; text-decoration: underline; }
    .user-bubble a { color: #bbdefb; }
    .bubble-sources {
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e8eaf0;
    }
    .bubble-time {
        font-size: 0.7rem;
        color: #9e9e9e;
        margin-top: 4px;
        padding: 0 4px;
    }
    .chat-row.user .bubble-time { text-align: right; }

    /* ── Welcome card ── */
    .welcome-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #1976d2;
    }
    .welcome-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: #1565c0;
    }
    .welcome-card p {
        color: #616161;
        font-size: 0.9rem;
        margin: 0 0 0.75rem 0;
    }
    .welcome-suggestion {
        display: inline-block;
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
        color: #424242;
        cursor: default;
        transition: background 0.2s ease;
    }
    .welcome-suggestion:hover {
        background: #e3f2fd;
        border-color: #90caf9;
        color: #1565c0;
    }

    /* ── Streaming message (st.chat_message) ── */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: none;
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
        margin-right: 15%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    /* Hide default avatar in streaming st.chat_message */
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessage"] .stAvatar {
        display: none;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.12);
    }

    /* ── Source chips (Material chip style) ── */
    .source-chip {
        display: inline-flex;
        align-items: center;
        background: #e3f2fd;
        border: none;
        border-radius: 16px;
        padding: 5px 12px;
        margin: 4px 4px 4px 0;
        font-size: 0.8rem;
        font-weight: 500;
        color: #1565c0;
        text-decoration: none;
        transition: background 0.2s ease, box-shadow 0.2s ease;
        line-height: 1.4;
    }
    .source-chip:hover {
        background: #bbdefb;
        box-shadow: 0 1px 4px rgba(21, 101, 192, 0.2);
        color: #0d47a1;
        text-decoration: none;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        font-weight: 600;
        font-size: 0.95rem;
        color: #424242;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sidebar-badge {
        background: #e8f5e9;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 500;
        color: #2e7d32;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* ── Material-style buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: none;
        letter-spacing: 0.01em;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        background: #1976d2;
        color: white;
        box-shadow: 0 1px 3px rgba(25, 118, 210, 0.3);
        transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.1s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1565c0;
        box-shadow: 0 4px 8px rgba(25, 118, 210, 0.35);
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(25, 118, 210, 0.3);
    }

    /* Second button (Clear Chat) as outlined variant */
    [data-testid="stSidebar"] .stButton:nth-of-type(2) > button {
        background: transparent;
        color: #d32f2f;
        border: 1px solid #ef9a9a;
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton:nth-of-type(2) > button:hover {
        background: #ffebee;
        box-shadow: none;
        transform: none;
    }

    /* ── Password input ── */
    [data-testid="stTextInput"] input {
        font-family: 'Inter', sans-serif;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        transition: border-color 0.2s ease;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.12);
    }

    /* ── Spinner/alerts ── */
    .stSpinner > div {
        border-top-color: #1976d2 !important;
    }
    [data-testid="stAlert"] {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #bdbdbd; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #9e9e9e; }
</style>
""", unsafe_allow_html=True)

# App bar
st.markdown("""
<div class="mui-appbar">
    <h1>IRS Tax Assistant</h1>
    <p>Answers grounded in official IRS.gov documentation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")

    # Show index stats
    import os
    raw_count = len([f for f in os.listdir("data/raw") if f.endswith(".json")]) if os.path.exists("data/raw") else 0
    if raw_count > 0:
        st.markdown(f'<div class="sidebar-badge">✅ {raw_count:,} documents indexed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-badge" style="background:#fef2f2;border-color:#fecaca;color:#991b1b;">⚠️ No data indexed yet</div>', unsafe_allow_html=True)

    if st.button("🔄 Re-index IRS Data", use_container_width=True):
        with st.spinner("Scraping IRS.gov..."):
            count = scrape_irs()
            st.success(f"Scraped {count} documents.")
        with st.spinner("Building index..."):
            chunks = build_index()
            st.success(f"Indexed {chunks} chunks.")
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.8rem;color:#9ca3af;text-align:center;'>"
        "Data source: <a href='https://www.irs.gov' target='_blank'>IRS.gov</a>"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='font-size:0.72rem;color:#b0b0b0;text-align:center;line-height:1.4;'>"
        "⚠️ This tool is for informational purposes only and does not constitute "
        "tax, legal, or financial advice. Consult a qualified tax professional "
        "for guidance on your specific situation."
        "</p>",
        unsafe_allow_html=True,
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources_html(sources: list[dict]) -> str:
    """Build HTML for source chips inside a bubble."""
    seen = set()
    chips = ""
    for src in sources:
        if src["source_url"] not in seen:
            title = src["title"] or src["source_url"].split("/")[-1]
            chips += f'<a class="source-chip" href="{src["source_url"]}" target="_blank">{title}</a>'
            seen.add(src["source_url"])
    return f'<div class="bubble-sources">{chips}</div>' if chips else ""


def render_bubble(role: str, content: str, sources: list[dict] | None = None):
    """Render a single chat bubble as custom HTML."""
    import markdown as md

    content_html = md.markdown(content, extensions=["fenced_code", "tables"])
    sources_block = render_sources_html(sources) if sources else ""

    if role == "user":
        html = f"""
        <div class="chat-row user">
            <div>
                <div class="chat-bubble user-bubble">{content_html}</div>
            </div>
            <div class="chat-avatar user-av">You</div>
        </div>"""
    else:
        html = f"""
        <div class="chat-row assistant">
            <div class="chat-avatar assistant-av">IRS</div>
            <div>
                <div class="chat-bubble assistant-bubble">{content_html}{sources_block}</div>
            </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# Welcome message when chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome to the IRS Tax Assistant</h3>
        <p>I can help answer tax questions using official IRS documentation.</p>
        <div>
            <span class="welcome-suggestion">When is the tax filing deadline?</span>
            <span class="welcome-suggestion">What is the Child Tax Credit?</span>
            <span class="welcome-suggestion">Forms for self-employment income?</span>
            <span class="welcome-suggestion">401(k) and IRA contribution limits?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display chat history as custom bubbles
for message in st.session_state.messages:
    render_bubble(message["role"], message["content"], message.get("sources"))

# Chat input
if prompt := st.chat_input("Ask a tax question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_bubble("user", prompt)

    # Use st.chat_message for streaming (needed for write_stream),
    # styled via CSS to look like an assistant bubble
    with st.chat_message("assistant", avatar="📋"):
        try:
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            sources, stream = ask(prompt, chat_history)
            answer = st.write_stream(stream)

            if sources:
                st.markdown(render_sources_html(sources), unsafe_allow_html=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Error: {e}. Make sure you've indexed IRS data first (use the sidebar button).")
