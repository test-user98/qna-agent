import os
import argparse
import re
import json
from typing import List, Dict, Any
from langchain.chains.summarize import load_summarize_chain
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import RetrievalQA
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

class DocumentQAAgent:
    def __init__(self, base_url: str, output_dir: str = 'output'):
        """
        Initialize the QA Agent with a base URL for documentation
        
        Args:
            base_url (str): Base URL of the help documentation
            output_dir (str): Directory to store processed documents and embeddings
        """
        self.base_url = base_url
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths for storing intermediate files
        self.scraped_urls_path = os.path.join(output_dir, 'scraped_urls.json')
        self.full_documents_path = os.path.join(output_dir, 'full_documents.json')
        
        # Initialize components
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(output_dir, 'chromadb'))
        self.collection = None
        
    def validate_url(self) -> bool:
        """
        Validate the input URL
        
        Returns:
            bool: Whether the URL is valid and accessible
        """
        try:
            response = requests.head(self.base_url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException:
            print(f"Error: Cannot access URL {self.base_url}")
            return False
    
    def extract_links(self, max_depth: int = 2, max_urls: int = 50) -> List[str]:
        """
        Extract valid documentation links from the base URL
        
        Returns:
            List[str]: List of unique URLs to scrape
        """
        def is_valid_help_url(url: str) -> bool:
            """Internal helper to validate documentation URLs"""
            parsed = urlparse(url)
            return (
                parsed.netloc == urlparse(self.base_url).netloc and
                not url.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif')) and
                not any(x in url.lower() for x in ['#', 'login', 'signup', 'account', 'get-started'])
            )
        
        def is_slack_help_url(url: str) -> bool:
            parsed = urlparse(url)
            return (
                parsed.netloc == 'slack.com' and
                not url.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif')) and
                not any(x in url for x in ['#', 'login', 'signup', 'account', 'get-started']) and
                ('/en' in url or '/articles/' in url)
            )
        
        def fetch_links(url: str) -> List[str]:
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                soup = BeautifulSoup(response.text, 'lxml')
                links = [urljoin(self.base_url, link['href']) for link in soup.find_all('a', href=True)]
                return links
            except Exception as e:
                print(f"Error fetching links from {url}: {e}")
                return []
        
        unique_urls = set()
        to_scrape = [(self.base_url, 0)]
        scraped_urls = set()
        
        while to_scrape and len(unique_urls) < max_urls:
            current_url, depth = to_scrape.pop(0)

            # Skip already scraped URLs or exceeding depth
            if current_url in scraped_urls or depth > max_depth:
                continue

            scraped_urls.add(current_url)
            new_links = fetch_links(current_url)

            for link in new_links:
                if is_valid_help_url(link) or is_slack_help_url(link):
                    if link not in unique_urls:
                        unique_urls.add(link)
                        to_scrape.append((link, depth + 1))

        print(f"Found {len(unique_urls)} unique URLs to scrape")
        return list(unique_urls)[:max_urls]


    def load_documents(self, urls: List[str]) -> List[Dict]:
        """
        Load and process documents from given URLs

        Args:
            urls (List[str]): List of URLs to scrape

        Returns:
            List[Dict]: Processed documents
        """
        documents = []
        scraped_urls = set()

        # Load existing scraped URLs
        if os.path.exists(self.scraped_urls_path):
            with open(self.scraped_urls_path, 'r') as f:
                try:
                    content = f.read().strip()
                    if content:
                        scraped_urls = set(json.loads(content))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {self.scraped_urls_path}. Initializing an empty set.")
                    scraped_urls = set()

        for url in urls:
            try:
                if url in scraped_urls:
                    continue

                response = requests.get(
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                    allow_redirects=True,
                )
                soup = BeautifulSoup(response.text, 'lxml')

                # Extract metadata from <script type="application/ld+json">
                metadata = {}
                json_ld_script = soup.find("script", type="application/ld+json")
                if json_ld_script:
                    try:
                        json_ld_data = json.loads(json_ld_script.string)
                        metadata['headline'] = json_ld_data.get('headline', 'No Headline')
                        metadata['description'] = json_ld_data.get('description', 'No Description')
                        metadata['articleBody'] = json_ld_data.get('articleBody', 'No Article Body')
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON-LD metadata from {url}")

                # Remove script, style, and navigation elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()

                # Extract main content
                main_content = soup.find(['main', 'article', 'div', 'body'])
                if not main_content:
                    main_content = soup

                # Clean text
                text = main_content.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text).strip()

                # Add metadata to document
                document = {
                    'page_content': text,
                    'metadata': {
                        'source': url,
                        'title': soup.title.string if soup.title else 'No Title',
                        **metadata  # Merge extracted metadata
                    },
                }

                documents.append(document)
                scraped_urls.add(url)
                print(f"Scraped: {url}")

            except Exception as e:
                print(f"Error scraping {url}: {e}")

        # Save scraped URLs
        with open(self.scraped_urls_path, 'w') as f:
            json.dump(list(scraped_urls), f)

        return documents


    def create_embeddings(self, documents: List[Dict]):
        """
        Create embeddings for documents and store in ChromaDB
        
        Args:
            documents (List[Dict]): List of processed documents
        """
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        langchain_docs = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents]
        chunks = text_splitter.split_documents(langchain_docs)
        
        # Create or get Chroma collection
        self.collection = self.chroma_client.get_or_create_collection("documentation_qa")
        
        # Process chunks and create embeddings
        for idx, chunk in enumerate(chunks):
            text = chunk.page_content
            metadata = chunk.metadata
            
            if not text:
                continue
            
            chunk_id = f"chunk_{idx}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Add to collection
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                documents=[text]
            )
        
        print(f"Created embeddings for {len(chunks)} chunks")
    

    def summarize_documents(self, documents: List[Document], question: str) -> str:

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Load Mistral model
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        # Create pipeline
        summarization_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=600
        )

        # Convert to LangChain LLM
        llm = HuggingFacePipeline(pipeline=summarization_pipeline)

        # Prepare context for summarization
        context = " ".join([chunk.page_content for chunk in chunks])
        
        # Create summarization prompt
        summarization_prompt = f"""Summarize the following text concisely and accurately and behave like an knowledgable question and answering agent, give short, brief and infrmative response.:

        {context}

        Summary:"""
        summary = llm(summarization_prompt)
        
        return summary.strip()

    def query(self, question: str, top_k: int = 3, summarize: bool=True) -> Dict[str, Any]:
        if not self.collection:
            raise ValueError("Embeddings not created. Run process_documentation first.")
        
        # Generate embedding for the question
        query_embedding = self.embedding_model.encode(question)
        
        # Perform semantic search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Process and format results
        if not results['documents'] or len(results['documents'][0]) == 0:
            return {
                'answer': "Sorry, I couldn't find any information related to your question.",
                'sources': []
            }
        
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
        
        # Prepare context
        context = " ".join([doc.page_content for doc in documents])
        sources = [doc.metadata.get('source', 'Unknown source') for doc in documents]
        
        # Optionally summarize documents
        if summarize and len(context) > 1000:
            context = self.summarize_documents(documents, question)
        
        # Load Mistral model
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        # Create pipeline
        qa_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=250
        )

        # Convert to LangChain LLM
        llm = HuggingFacePipeline(pipeline=qa_pipeline)

        # Create prompt with context and question
        prompt = f"""Using only the following context, answer the question precisely and accurately.  behave like an knowledgable question and answering agent, give short, brief and infrmative response.:

        Context:
        {context}

        Question: {question}

        Answer:"""

        # Generate response
        answer = llm(prompt)
        
        return {
            "answer": answer.strip(),
            "sources": sources
        }

    def process_documentation(self):
        """
        Full pipeline to process documentation
        """
        if not self.validate_url():
            print("Invalid or unreachable URL")
            return False
        
        # Extract and process links
        urls_to_scrape = self.extract_links()
        if not urls_to_scrape:
            print("No valid URLs found")
            return False
        
        # Load documents
        print("URLS TO scrapre", urls_to_scrape)
        documents = self.load_documents(urls_to_scrape)
        if not documents:
            print("No documents scraped")
            return False
        
        # Create embeddings
        self.create_embeddings(documents)
        
        return True

def main():
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Documentation QA Agent")
    parser.add_argument("--url", required=True, help="Base URL of help documentation")
    args = parser.parse_args()
    
    # Initialize and process documentation
    agent = DocumentQAAgent(args.url)

    if not agent.process_documentation():
        print("Failed to process documentation")
        return
    
    # Interactive QA loop
    print(f"\n🤖 Documentation QA Agent initialized for {args.url}")
    print("Type your questions. Enter 'quit' to exit.")
    
    while True:
        try:
            question = input("\n> ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question.strip():
                continue
            
            # Query the agent
            result = agent.query(question)
            
            print("\n📖 Answer:", result['answer'])
            print("\n🔗 Sources:")
            for source in result['sources']:
                print(f"- {source}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()