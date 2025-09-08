import os
import time
import requests
import fitz  # PyMuPDF
from tqdm import tqdm
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import utils as Utils


def pdf_to_text(source: str) -> str:
    """
    Convert a PDF (from URL or local file) into plain text.
    """
    try:
        if source.startswith("http"):
            response = requests.get(source)
            document = fitz.open(stream=response.content, filetype="pdf")
        else:
            document = fitz.open(source)

        text = "".join(document.load_page(i).get_text() for i in range(len(document)))
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def split_text_into_sections(text: str, min_chars: int = 1500) -> list:
    """
    Split text into sections of at least `min_chars` characters.
    """
    paragraphs = text.split("\n")
    sections = []
    current_section = ""
    current_length = 0

    for para in paragraphs:
        if current_length + len(para) + 2 <= min_chars:
            current_section += para + "\n\n"
            current_length += len(para) + 2
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = para + "\n\n"
            current_length = len(para) + 2

    if current_section:
        sections.append(current_section.strip())

    return sections


def embed_text_in_chromadb(
    text: str,
    document_name: str,
    document_description: str,
    persist_directory: str = Utils.DB_FOLDER,
    batch_size: int = 5,
    delay: float = 1.5
):
    """
    Embed text into ChromaDB in batches, with retries for failed embeddings.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in environment variables.")

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    sections = split_text_into_sections(text)
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("indian_laws_collection")

    start_idx = collection.count()
    ids = [str(i) for i in range(start_idx, start_idx + len(sections))]
    metadatas = [
        {"name": document_name, "description": document_description, "section": i + 1}
        for i in range(len(sections))
    ]

    all_embeddings = []

    for i in tqdm(range(0, len(sections), batch_size), desc="Embedding batches"):
        batch = sections[i:i + batch_size]
        success = False
        retries = 3

        while not success and retries > 0:
            try:
                batch_embeddings = embeddings_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                success = True
            except Exception as e:
                retries -= 1
                print(f"Batch failed. Retrying ({3 - retries}/3)... Error: {e}")
                time.sleep(delay * 2)

        if not success:
            print(f"Skipping batch {i // batch_size} after 3 failed attempts.")
            all_embeddings.extend([[0.0] * embeddings_model.embedding_dim] * len(batch))

        time.sleep(delay)

    # Add to ChromaDB
    for i in tqdm(range(0, len(sections), batch_size), desc="Adding to ChromaDB"):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=sections[i:i + batch_size],
            embeddings=all_embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
        time.sleep(delay)

    print(f"Successfully added {len(sections)} sections to ChromaDB.")
    print(f"Collection now contains {collection.count()} documents.")
