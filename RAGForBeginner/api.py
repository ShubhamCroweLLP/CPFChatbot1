from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ingestion_pipeline import main as run_ingestion

load_dotenv()

app = FastAPI(title="RAG API")

# --- Shared state ---
persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
model = ChatOpenAI(model="gpt-4o")

# Per-session chat history (simple in-memory; for production use a DB)
sessions: dict[str, list] = {}


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ingest")
def ingest_documents():
    """Re-run the ingestion pipeline to load/update documents."""
    try:
        run_ingestion()
        return {"status": "ok", "message": "Ingestion complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    chat_history = sessions.setdefault(req.session_id, [])

    # Rewrite question if there's history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {req.question}")
        ]
        search_question = model.invoke(messages).content.strip()
    else:
        search_question = req.question

    # Retrieve relevant docs
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    combined_input = f"""Based on the following documents, please answer this question: {req.question}

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

    # Remember conversation
    chat_history.append(HumanMessage(content=req.question))
    chat_history.append(AIMessage(content=answer))

    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    return AnswerResponse(answer=answer, sources=sources)


@app.post("/clear")
def clear_history(session_id: str = "default"):
    sessions.pop(session_id, None)
    return {"status": "ok", "message": "Chat history cleared."}
