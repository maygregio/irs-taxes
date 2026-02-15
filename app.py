# app.py
import streamlit as st
from src.chain import ask
from src.scraper import scrape_irs
from src.indexer import build_index

st.set_page_config(page_title="IRS Tax Assistant", page_icon="📋", layout="centered")
st.title("IRS Tax Assistant")
st.caption("Answers grounded in official IRS.gov documentation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Re-index IRS Data"):
        with st.spinner("Scraping IRS.gov..."):
            count = scrape_irs()
            st.success(f"Scraped {count} documents.")
        with st.spinner("Building index..."):
            chunks = build_index()
            st.success(f"Indexed {chunks} chunks.")

    st.markdown("---")
    st.markdown("Data source: [IRS.gov](https://www.irs.gov)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- [{src['title']}]({src['source_url']})")

# Chat input
if prompt := st.chat_input("Ask a tax question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching IRS documents..."):
            try:
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude current question
                ]
                answer, sources = ask(prompt, chat_history)
                st.markdown(answer)

                if sources:
                    with st.expander("📄 Sources"):
                        seen = set()
                        for src in sources:
                            if src["source_url"] not in seen:
                                st.markdown(f"- [{src['title']}]({src['source_url']})")
                                seen.add(src["source_url"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Error: {e}. Make sure you've indexed IRS data first (use the sidebar button).")
