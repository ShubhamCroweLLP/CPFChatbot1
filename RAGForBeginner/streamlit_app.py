import os
import uuid
import streamlit as st
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# --- Init ---
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)
model = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


@st.cache_resource
def build_vectorstore():
    """Load docs and build FAISS index in memory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "docs")
    if not os.path.exists(docs_path):
        st.error(f"No '{docs_path}' directory found.")
        st.stop()
    loader = DirectoryLoader(
        path=docs_path, glob="**/*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        st.error("No .txt files found in docs/")
        st.stop()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


# --- Helpers for multi-conversation management ---
def new_conversation():
    """Create a new conversation and set it as active."""
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New Chat",
        "messages": [],
        "chat_history": [],
    }
    st.session_state.active_conversation = conv_id
    return conv_id


def get_active_conv():
    """Return the active conversation dict."""
    return st.session_state.conversations[st.session_state.active_conversation]


# --- UI ---
st.set_page_config(page_title="CPF Assist", page_icon="🤖")
st.title("🤖 CPF Assist")

# Initialize conversation store
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_conversation" not in st.session_state:
    new_conversation()
# Ensure active conversation still exists
if st.session_state.active_conversation not in st.session_state.conversations:
    new_conversation()

with st.sidebar:
    st.header("💬 Chat History")

    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        new_conversation()
        st.rerun()

    st.divider()

    # List all conversations (newest first)
    for conv_id in reversed(list(st.session_state.conversations.keys())):
        conv = st.session_state.conversations[conv_id]
        is_active = conv_id == st.session_state.active_conversation
        label = conv["title"]
        # Truncate long titles
        if len(label) > 30:
            label = label[:27] + "..."

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                f"{'▶ ' if is_active else ''}{label}",
                key=f"conv_{conv_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_conversation = conv_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{conv_id}"):
                del st.session_state.conversations[conv_id]
                if st.session_state.active_conversation == conv_id:
                    if st.session_state.conversations:
                        st.session_state.active_conversation = list(
                            st.session_state.conversations.keys()
                        )[-1]
                    else:
                        new_conversation()
                st.rerun()

db = build_vectorstore()
conv = get_active_conv()

# Display chat history for the active conversation
for msg in conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Auto-title: use the first user message as the conversation title
    if conv["title"] == "New Chat":
        conv["title"] = prompt[:50] if len(prompt) <= 50 else prompt[:47] + "..."

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history = conv["chat_history"]

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

            st.markdown(answer)

            # Update history
            conv["messages"].append({"role": "assistant", "content": answer})
            conv["chat_history"].append(HumanMessage(content=prompt))
            conv["chat_history"].append(AIMessage(content=answer))
            st.rerun()
