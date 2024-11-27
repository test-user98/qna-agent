import os
import re
import json
import hashlib
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
import chromadb
from sentence_transformers import SentenceTransformer

def is_slack_help_url(url):
    parsed = urlparse(url)
    
    return (
        parsed.netloc == 'slack.com' and 
        not url.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif')) and
        not any(x in url for x in ['#', 'login', 'signup', 'account', 'get-started']) and
        
        ('/en' in url or '/articles/' in url)
    )

def custom_link_extractor(base_url, scraped_urls_path):
    def is_valid_help_url(url):
        parsed = urlparse(url)
        return (
            parsed.netloc == 'help.slack.com' and 
            not url.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif')) and
            not any(x in url for x in ['#', 'login', 'signup', 'account', 'get-started'])
        )
    
    unique_urls = set()
    
    try:
        scraped_urls = set()
        if os.path.exists(scraped_urls_path):
            with open(scraped_urls_path, 'r') as f:
                scraped_urls = set(json.load(f))
        
        response = requests.get(base_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'lxml')
        
        for link in soup.find_all('a', href=True):
            full_url = urljoin(base_url, link['href'])
            
            if full_url in scraped_urls:
                continue
            
            if 'slack.com' in base_url:
                if is_slack_help_url(full_url) and full_url not in unique_urls:
                    unique_urls.add(full_url)

            else:
                if is_valid_help_url(full_url) and full_url not in unique_urls:
                    unique_urls.add(full_url)
        
        print(f"Found {len(unique_urls)} unique URLs to scrape")
        return list(unique_urls)
    
    except Exception as e:
        print(f"Error extracting links: {e}")
        return []

def custom_document_loader(urls, scraped_urls_path, existing_documents_path=None):
    documents = []
    
    scraped_urls = set()
    if os.path.exists(scraped_urls_path):
        with open(scraped_urls_path, 'r') as f:
            scraped_urls = set(json.load(f))
    
    existing_documents = []
    if existing_documents_path and os.path.exists(existing_documents_path):
        with open(existing_documents_path, 'r', encoding='utf-8') as f:
            existing_documents = json.load(f)
    
    for url in urls:
        try:
            if url in scraped_urls:
                continue
            
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'lxml')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            main_content = soup.find(['main', 'article', 'div', 'body'])
            if not main_content:
                main_content = soup
            
            text = main_content.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            
            document = {
                'page_content': text,
                'metadata': {
                    'source': url,
                    'title': soup.title.string if soup.title else 'No Title',
                }
            }
            
            documents.append(document)
            scraped_urls.add(url)
            print(f"Scraped: {url}")
        
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    with open(scraped_urls_path, 'w') as f:
        json.dump(list(scraped_urls), f)
    
    documents.extend(existing_documents)
    return documents

def save_documents(documents, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    full_docs_path = os.path.join(output_dir, 'full_documents.json')
    
    with open(full_docs_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    from langchain_core.documents import Document
    langchain_docs = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents]
    
    chunks = text_splitter.split_documents(langchain_docs)
    
    chunks_path = os.path.join(output_dir, 'document_chunks.json')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump([
            {
                'page_content': chunk.page_content,
                'metadata': chunk.metadata
            } 
            for chunk in chunks
        ], f, indent=2, ensure_ascii=False)
    
    return documents, chunks

def create_embeddings(chunks, output_dir='output'):
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    client = chromadb.PersistentClient(path=os.path.join(output_dir, 'chromadb'))
    collection = client.get_or_create_collection("slack_help_docs")

    for idx, chunk in enumerate(chunks):
        # Extract content and metadata from the Document object
        text = getattr(chunk, "page_content", "").strip()
        metadata = getattr(chunk, "metadata", {})

        if not text:
            print(f"Skipping chunk {idx}: No content found.")
            continue

        # Generate a unique chunk ID
        chunk_id = metadata.get("id", f"chunk_{idx}")

        # Generate embedding
        embedding = model.encode(text)

        if embedding is None or len(embedding) == 0:
            raise ValueError(f"Embedding generation failed for the given text: {text}")
        
        # Check if the chunk_id already exists in the collection
        existing_ids = collection.get(ids=[chunk_id]).get("ids", [])
        if chunk_id in existing_ids:
            print(f"Chunk ID {chunk_id} already exists. Overwriting...")
            collection.delete(ids=[chunk_id])  # Remove the existing chunk
        
        # Add the new chunk
        # print("***********************embedding***********************", chunk_id, embedding.tolist())
        # return 0
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[text]
        )
    
    print(f"Created embeddings for {len(chunks)} chunks")
    return collection

def main():
    base_url = "https://help.slack.com"
    output_dir = "output"
    scraped_urls_path = os.path.join(output_dir, 'scraped_urls.json')
    full_documents_path = os.path.join(output_dir, 'full_documents.json')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Scrape URLs
    urls_to_scrape = custom_link_extractor(base_url, scraped_urls_path)
    urls_to_scrape = urls_to_scrape[:50]  # Limit to 50 URLs
    
    # Step 2: Load documents
    documents = custom_document_loader(urls_to_scrape, scraped_urls_path, full_documents_path)
    
    # Step 3: Save documents and split into chunks
    documents, chunks = save_documents(documents)
    
    # Step 4: Create embeddings and save to the collection
    collection = create_embeddings(chunks)

    # Step 5: Retrieve a sample chunk for inspection
    sample_chunk = chunks[0]

    chunk_id = sample_chunk.metadata.get("id", f"chunk_2")  # Use fallback id "chunk_0" for testing

    try:
        sample_embedding = collection.get(ids=[chunk_id], include=['embeddings', 'documents', 'metadatas'])
        print("\nSample Chunk Embedding:", sample_embedding)
    except Exception as e:
        print(f"Error retrieving embedding for chunk_id {chunk_id}: {e}")

    # Final stats
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

if __name__ == "__main__":
    main()
