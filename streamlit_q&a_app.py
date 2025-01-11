#chromadb is more suitable for smaller size files. You can use
# an online vector store using Pinecone for larger projects

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
import os
import chromadb.api 
 

# this was need 
chromadb.api.client.SharedSystemClient.clear_system_cache()

# function to load pdf, text and docx files. 
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# function to create vector embeddings from the chunks
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings() # initialize an instance of this clas
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store 

# ask and get function
# it will return the k most similar chunks in the simlarity search
# the llm will take the k chunks and create the answerr in NL. Higher the k value, the higher the value you will pay as you will use more chunks
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state: # if the history exists, we will delete it
        del st.session_state['history']

if __name__ == "__main__":
    import os
    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    #for the img, using just img on its own did not work. Needed to refereence the folder itself
    st.image('.\streamlit_q&a_app\img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        #chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # this is a button for add data. When the button is clicked, the chunks will be embedded to numeric vectors and stored in Chroma vector store. 
        add_data = st.button('Add Data', on_click=clear_history)

# if the user uploaded a file and clickec add_data, add a spinner which is a widget that temporarily displays a message while executing a block of code
# the uploaed file is saved as BytesIO file in python memory on RAM. 
# we will copy the file thats in RAM to a disc locally. 
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file...'):
                bytes_data = uploaded_file.read() # read the contents of the file in binary
                file_name = os.path.join('./', uploaded_file.name) #copy the file to the current directory
                with open(file_name, 'wb') as f: #in write and binary mode
                    f.write(bytes_data)

                data = load_document(file_name) # load the data. Next we will chunk
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                # # display embedding cost
                # tokens, embedding_cost = calculate_embedding_cost(chunks)
                # st.write(f'Embedding cost: ${embedding_cost: .4f}')

            # embed the chunks
            # we will save the vector store between page reloads, as we dont want to chunk and load the file and embed the chunks each time the user interacts with the widgets.
            # we will save the vector store in the session_state as vs. 
            vector_store = create_embeddings(chunks) 

            st.session_state.vs = vector_store
            st.success('File uploaded, chunked and embedded successfully')

    # we are done with the sider bar. We will display the main pagem, where we will ask quesionts
    # and receieve answers.
    # we will check to see if the question already exists in the session state (vs) and if so, we will load from the session state 
    # 
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            #st.write(f'k: {k}') # this is used for debugging, which we can comment out when everything works as expected
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('LLM Answer:', value =answer)

        # currently, we can only see the last question and answer. We will save each question and answer to the session history and display them all
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}' #the current questions and answers. We are adding this before the history, as we want it displayed first
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history # we will use this to display the chat histroy in a text area
            st.text_area(label='Chat History', value=h, key='history', height=400) #adding key=history means we will store in the session state with a value named history

