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

# Custom styling
st.markdown("""
<style>
    /* Clean header area */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
    }

    /* Style source links */
    .source-chip {
        display: inline-block;
        background: #f0f4ff;
        border: 1px solid #d0d9f0;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.82rem;
        color: #1a56db;
        text-decoration: none;
    }
    .source-chip:hover {
        background: #dbe4ff;
    }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        padding-top: 1.5rem;
    }
    .sidebar-badge {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        font-size: 0.85rem;
        color: #065f46;
        margin-bottom: 1rem;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat message spacing */
    [data-testid="stChatMessage"] {
        padding: 0.75rem 1rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📋 IRS Tax Assistant</h1>
    <p>Answers grounded in official IRS.gov documentation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

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

# Welcome message when chat is empty
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="📋"):
        st.markdown(
            "👋 **Welcome!** I can help answer tax questions using official IRS documentation.\n\n"
            "Try asking things like:\n"
            "- *When is the tax filing deadline?*\n"
            "- *What is the Child Tax Credit and who qualifies?*\n"
            "- *What forms do I need for self-employment income?*\n"
            "- *What are the contribution limits for 401(k)s and IRAs?*"
        )

# Display chat history
for message in st.session_state.messages:
    avatar = "📋" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            sources_html = "<div style='margin-top:8px;'>"
            seen = set()
            for src in message["sources"]:
                if src["source_url"] not in seen:
                    title = src["title"] or src["source_url"].split("/")[-1]
                    sources_html += f'<a class="source-chip" href="{src["source_url"]}" target="_blank">📄 {title}</a>'
                    seen.add(src["source_url"])
            sources_html += "</div>"
            st.markdown(sources_html, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a tax question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📋"):
        try:
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            sources, stream = ask(prompt, chat_history)
            answer = st.write_stream(stream)

            if sources:
                sources_html = "<div style='margin-top:8px;'>"
                seen = set()
                for src in sources:
                    if src["source_url"] not in seen:
                        title = src["title"] or src["source_url"].split("/")[-1]
                        sources_html += f'<a class="source-chip" href="{src["source_url"]}" target="_blank">📄 {title}</a>'
                        seen.add(src["source_url"])
                sources_html += "</div>"
                st.markdown(sources_html, unsafe_allow_html=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Error: {e}. Make sure you've indexed IRS data first (use the sidebar button).")
