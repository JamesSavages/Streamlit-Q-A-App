# Streamlit Question & Answering App

### Project Description
Welcome to our project! This is a question and answering app in Python and Streamlit, as well as the Chroma Vector Database. It all allows a user to easily load a file (txt, pdf, docx) and ask questions related to that file. The LLM will output answers for the questions asked about this file loaded. Streamlit is used to craft a front end UI, which provides an intuitive, easy to use application.  

This project Retrieval-Augmented Generation (RAG) which is a is a technique that combines language models (OpenAI in this case) with information retrieval mechanisms to improve text generation. Instead of relying only on data learned by the model, RAG retrieves relevant information from external sources, such as documents or databases, to provide more accurate and reliable answers. RAG heavily used in the Generative AI space, due to its capcabilities to:

1. Reduce hallucinations -  RAG uses retrieval systems to access verified information from external sources in real time, reducing the likelihood of hallucinations and increasing the reliability of responses
2. Domain knowledge -  RAG enables the integration of domain-specific data from external sources, improving the accuracy and relevance of responses
3. Training data cut off -  RAG can search for updated information in databases or on the internet, ensuring that the answers are always up-to-date and relevant

### Steps and tasks:

#### 1. Loading the Document
Load_document - A function that consists of the file types that can be loaded. As of now, this is just set to pdf, docx, and txt file types.

#### 2. Splitting Data into Chunks
chunk_data - function for chunking the data. This involves breaking the data into smaller pieces, making the data easier to index. It consists of a pipeline for ingesting data from a source and indexing it. The typical method is to convert all private data into embeddings stored in a vector database. 

#### 3. Create Embeddings and store
create_embeddings - a function using OpenAIEmbeddings to convert the chunks to vectors and store them in the Chroma Vector Store. Chroma is a popular choice, along with other options such as Pinecone, QDrant and Milvus.   

#### 4. Question & Answering
ask_and_get_answer - function to retrieve data from the vector store, which is then passed to the LLM. Similarity Search is applied, which compares the numerical vectors (embeddings) of the query and the 
documents to find the most similar ones via KNN.
