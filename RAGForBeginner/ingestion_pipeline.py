import os
import base64
import glob
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp")
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    txt_loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    # Load all .pdf files from the docs directory
    pdf_loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    
    documents = txt_loader.load() + pdf_loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt or .pdf files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents


def _get_vision_model():
    """Create a GPT-4o vision model instance for describing images."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _describe_image(vision_model, image_path):
    """Use GPT-4o vision to generate a text description of an image."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail. Include all visible text, labels, diagrams, UI elements, and any information that would be useful for someone searching for this content."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
        ]
    )
    response = vision_model.invoke([message])
    return response.content


def load_images(docs_path="docs"):
    """Load images from docs directory, describe them with GPT-4o vision, and return as Documents."""
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(docs_path, "**", ext), recursive=True))

    if not image_files:
        print("No image files found.")
        return []

    print(f"Found {len(image_files)} image(s). Describing with GPT-4o vision...")
    vision_model = _get_vision_model()
    documents = []

    for img_path in image_files:
        try:
            print(f"  📷 Processing: {os.path.basename(img_path)}")
            description = _describe_image(vision_model, img_path)
            doc = Document(
                page_content=f"[Image: {os.path.basename(img_path)}]\n{description}",
                metadata={"source": img_path, "type": "image"}
            )
            documents.append(doc)
        except Exception as e:
            print(f"  ⚠️ Failed to process {img_path}: {e}")

    print(f"Successfully processed {len(documents)} image(s).")
    return documents


def load_videos(docs_path="docs"):
    """Load videos from docs directory, extract key frames, describe them with GPT-4o vision."""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(docs_path, "**", ext), recursive=True))

    if not video_files:
        print("No video files found.")
        return []

    try:
        import cv2
    except ImportError:
        print("⚠️ opencv-python-headless not installed. Skipping video processing.")
        return []

    print(f"Found {len(video_files)} video(s). Extracting frames and describing...")
    vision_model = _get_vision_model()
    documents = []

    for video_path in video_files:
        try:
            print(f"  🎥 Processing: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Extract 1 frame every 30 seconds, max 10 frames
            interval = max(30, duration / 10) if duration > 0 else 30
            frame_times = []
            t = 0
            while t < duration:
                frame_times.append(t)
                t += interval
                if len(frame_times) >= 10:
                    break

            frame_descriptions = []
            for t in frame_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue
                _, buffer = cv2.imencode(".jpg", frame)
                image_data = base64.b64encode(buffer).decode("utf-8")

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": f"This is a frame from a video at {t:.0f}s. Describe what you see in detail, including any text, UI elements, diagrams, or actions shown."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ]
                )
                resp = vision_model.invoke([message])
                frame_descriptions.append(f"[Frame at {t:.0f}s]: {resp.content}")

            cap.release()

            if frame_descriptions:
                full_description = f"[Video: {os.path.basename(video_path)} | Duration: {duration:.0f}s]\n" + "\n\n".join(frame_descriptions)
                doc = Document(
                    page_content=full_description,
                    metadata={"source": video_path, "type": "video", "duration_seconds": duration}
                )
                documents.append(doc)

        except Exception as e:
            print(f"  ⚠️ Failed to process {video_path}: {e}")

    print(f"Successfully processed {len(documents)} video(s).")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = AzureOpenAIEmbeddings(azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(base_dir, "docs")
    persistent_directory = os.path.join(base_dir, "db", "chroma_db")
    
    # Always remove old vector store and rebuild
    if os.path.exists(persistent_directory):
        import shutil
        print("🗑️ Removing existing vector store to rebuild...")
        shutil.rmtree(persistent_directory, ignore_errors=True)
    
    print("Initializing vector store...\n")
    
    # Step 1: Load documents (text, PDF, images, videos)
    documents = load_documents(docs_path)
    documents += load_images(docs_path)
    documents += load_videos(docs_path)
    print(f"\nTotal documents loaded: {len(documents)}")

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()




# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]
