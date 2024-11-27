import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader,RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
import re
import getpass
from langchain_chroma import Chroma
import json  

load_dotenv()

# os.environ['USER_AGENT'] = 'myagent'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env file")

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_test")

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def save_documents(documents, output_dir='output'):
    """
    Save scraped documents to JSON files
    
    Args:
        documents (list): List of documents to save
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full documents
    full_docs_path = os.path.join(output_dir, 'full_documents.json')
    with open(full_docs_path, 'w', encoding='utf-8') as f:
        json.dump([
            {
                'page_content': doc.page_content[:500] + '...' if len(doc.page_content) > 500 else doc.page_content,
                'metadata': doc.metadata
            } 
            for doc in documents
        ], f, indent=2, ensure_ascii=False)
    
    # Save chunk details
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    chunks_path = os.path.join(output_dir, 'document_chunks.json')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump([
            {
                'page_content': chunk.page_content,
                'metadata': chunk.metadata
            } 
            for chunk in docs
        ], f, indent=2, ensure_ascii=False)
    
    print(f"Total documents scraped: {len(documents)}")
    print(f"Total document chunks: {len(docs)}")
    print(f"Full documents saved to: {full_docs_path}")
    print(f"Document chunks saved to: {chunks_path}")
    
    return docs
def scrape_and_embed_docs(base_url: str, max_depth: int = 3):
    """
    Scrape documentation from a base URL and create embeddings
    
    Args:
        base_url (str): Base URL to start scraping
        max_depth (int): Maximum depth for recursive URL crawling
    """
    # Set up user agent to avoid potential blocking
    os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    

    loader = RecursiveUrlLoader(
        base_url, 
        max_depth=max_depth, 
        extractor=bs4_extractor,
        prevent_outside=True  
    )
    
    # Load documents
    print(f"Starting to scrape documents from {base_url}")
    documents = loader.load()
    print(f"Scraped {len(documents)} documents")
    
    # Split documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # docs = text_splitter.split_documents(documents)
    # print(f"Split into {len(docs)} document chunks")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full documents
    full_docs_path = os.path.join(output_dir, 'full_documents.json')
    with open(full_docs_path, 'w', encoding='utf-8') as f:
        json.dump([
            {
                'page_content': doc.page_content[:500] + '...' if len(doc.page_content) > 500 else doc.page_content,
                'metadata': doc.metadata
            } 
            for doc in documents
        ], f, indent=2, ensure_ascii=False)
    
    # Save chunk details
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    chunks_path = os.path.join(output_dir, 'document_chunks.json')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump([
            {
                'page_content': chunk.page_content,
                'metadata': chunk.metadata
            } 
            for chunk in docs
        ], f, indent=2, ensure_ascii=False)
    
    print(f"Total documents scraped: {len(documents)}")
    print(f"Total document chunks: {len(docs)}")
    print(f"Full documents saved to: {full_docs_path}")
    print(f"Document chunks saved to: {chunks_path}")
    # Create embeddings
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # # Create or load Chroma vector store
    # try:
    #     db = Chroma.from_documents(
    #         docs, 
    #         embeddings, 
    #         persist_directory=persistent_directory
    #     )
    #     print(f"Successfully created vector store in {persistent_directory}")
    # except Exception as e:
    #     print(f"Error creating vector store: {e}")
    #     return None
    
    return 0



def main():
    # Scrape Slack help documentation
    vector_db = scrape_and_embed_docs("https://help.slack.com")
    
    # if vector_db:
    #     # Create a retriever
    #     retriever = vector_db.as_retriever(
    #         search_type="similarity",
    #         search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
    #     )
        
    #     # Example query
    #     query = "how to get started with Slack?"
    #     print("\n--- Retrieving documents for query ---")
    #     relevant_docs = retriever.invoke(query)
        
    #     print(f"\nRetrieved {len(relevant_docs)} relevant documents:")
    #     for i, doc in enumerate(relevant_docs, 1):
    #         print(f"\nDocument {i}:")
    #         print(doc.page_content[:500] + "...")  # Print first 500 characters
    #         print(f"Source: {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()



# loader = RecursiveUrlLoader("https://help.slack.com",max_depth=3,extractor=bs4_extractor)
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# Display information about the split documents
# print("\n--- Document Chunks Information ---")
# print(f"Number of document chunks: {len(docs)}")
# print(f"Sample chunk:\n{docs[0].page_content}\n")


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# print(embeddings)

# if not os.path.exists(persistent_directory):
#     print(f"\n--- Creating vector store in {persistent_directory} ---")
#     db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
#     print(f"--- Finished creating vector store in {persistent_directory} ---")
# else:
#     print(f"Vector store {persistent_directory} already exists. No need to initialize.")
#     db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3},
# )

# print(retriever)

# # Define the user's question
# query = "how can a new user start with slack?"

# # Retrieve relevant documents based on the query
# relevant_docs = retriever.invoke(query)
# # print(relevant_docs)
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# # print(loader.load()[0].page_content)


