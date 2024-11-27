

# Agent-Based System for Response Generation

## Features

The Agent-Based System is designed to provide intelligent, context-aware responses based on input data. The system leverages various AI and data-driven techniques to analyze and generate responses that are relevant, accurate, and user-specific. Below are some key features of the agent:

We are using below models:
- To create embedding: sentence-transformers/all-MiniLM-L6-v2
- To generate and summarize response: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- using chroma vector db to store embeddings
- Also using cache to store the scraped urls, recorsively scrapping them.
- added functionality to exclude auth, login and filter and process the html doc to geenrate a json of document in belo format:

{
  page_content: <content>
  metadata: {
    source: <data>
    title: <tile>
    }
}

- we are creating chunks from the raw documents,
- defult recursive scraping is with depth: 3
- these chunk are used to create embeddings.




---

## Mapping Diagram: Response Generation Process

The following diagram outlines how the system processes an input and generates a response.

```plaintext
            +---------------------+        +-------------------+       
            |   User Input Data   | -----> |   generate embeddings  | 
            +---------------------+        +-------------------+       
                                                  |
                                                  v                       
                                                                  
                                       +------------------++------------------+           +----------------+
                                       |find similarly emebddings (3) |        -----> |   summarize and feed to model  -------> FINAL RESPONSE.   
                                      +------------------++------------------+           +----------------+
                                                             
                                                   
```

### Explanation:
1. **User Input**: The agent receives input data from the user, which can be a question, statement, or command.
2. **Data Preprocessing**: The input data is processed to extract meaningful information, such as entities or key phrases, to understand the user's intent.
3. **Context Extraction**: The system extracts the context from the processed data, considering past interactions or session history.
4. **Model Inference**: Using the extracted context, the system generates a response byquerying an Open source LLM (mistral or Llama)
5. **Final Response**: The final response is generated based on the model’s inference, formatted appropriately, and sent back to the user.

---

## Code Flow

1. **Input Handling**:
   - The system begins by receiving user input through the front-end interface or an API endpoint.
   
2. **Preprocessing**:
   - The input is passed through a preprocessing module that cleans and normalizes it for better understanding.
   - The input data might be tokenized, stopwords removed, and key phrases extracted.
   
3. **Intent Recognition**:
   - The system applies an NLP model or rule-based system to identify the user’s intent. This can include classifying the type of query (e.g., question, command).
   
4. **Context Management**:
   - The context manager tracks the conversation flow and stores relevant context information from previous interactions to provide more accurate and personalized responses.
   
5. **Response Generation**:
   - Based on the identified intent and context, the system queries a response model (like GPT, BERT, or custom model) to generate a relevant response.
   - If the model is not confident in the answer, fallback mechanisms are triggered (e.g., error message or clarification question).
   
6. **Post-processing**:
   - The generated response is formatted and tailored to the user’s input, ensuring it is clear and contextually appropriate.
   
7. **Return Response**:
   - The response is then returned to the user via the appropriate channel, be it a web interface, chat, or API.

---

## Setup and Installation

### Prerequisites:
- Python 3.7+
- Required libraries (e.g., `transformers`, `nltk`, `flask`)

### Installation Steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/test-user98/qna-agent
   ```

2. Navigate into the project directory:
   ```bash
   cd qna-agent
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the system:
   ```bash
   python document_qa.py
   ```
