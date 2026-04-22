import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CPF Chatbot", page_icon="🤖")
st.title("🤖 CPF Chatbot")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Re-ingest Documents"):
        with st.spinner("Running ingestion pipeline..."):
            resp = requests.post(f"{API_URL}/ingest")
            if resp.ok:
                st.success("Ingestion complete!")
            else:
                st.error(resp.json().get("detail", "Ingestion failed."))

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        requests.post(f"{API_URL}/clear")
        st.rerun()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": prompt},
            )
            if resp.ok:
                data = resp.json()
                answer = data["answer"]
                sources = data["sources"]
                full = f"{answer}\n\n📄 **Sources:** {', '.join(sources)}"
                st.markdown(full)
                st.session_state.messages.append({"role": "assistant", "content": full})
            else:
                st.error("Failed to get a response from the API.")
