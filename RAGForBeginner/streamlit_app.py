import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Init ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4o")


@st.cache_resource
def build_vectorstore():
    """Load docs and build FAISS index in memory."""
    docs_path = "docs"
    if not os.path.exists(docs_path):
        st.error(f"No '{docs_path}' directory found.")
        st.stop()
    loader = DirectoryLoader(
        path=docs_path, glob="*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        st.error("No .txt files found in docs/")
        st.stop()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


db = build_vectorstore()

# --- UI ---
st.set_page_config(page_title="CPF Chatbot", page_icon="🤖")
st.title("🤖 CPF Chatbot")

with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Re-ingest Documents"):
        with st.spinner("Running ingestion pipeline..."):
            st.cache_resource.clear()
            st.success("Ingestion complete!")
            st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history = st.session_state.chat_history

            # Rewrite question if there's history
            if chat_history:
                rewrite_messages = [
                    SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
                ] + chat_history + [
                    HumanMessage(content=f"New question: {prompt}")
                ]
                search_question = model.invoke(rewrite_messages).content.strip()
            else:
                search_question = prompt

            # Retrieve docs
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(search_question)

            combined_input = f"""Based on the following documents, please answer this question: {prompt}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
            ] + chat_history + [
                HumanMessage(content=combined_input)
            ]

            result = model.invoke(messages)
            answer = result.content
            sources = list({doc.metadata.get("source", "unknown") for doc in docs})
            full = f"{answer}\n\n📄 **Sources:** {', '.join(sources)}"

            st.markdown(full)

            # Update history
            st.session_state.messages.append({"role": "assistant", "content": full})
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
