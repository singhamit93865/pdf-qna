import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain import OpenAI, ConversationChain

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set the title of the web app
st.title('PDF Question Answering Web App')

def create_conversation_chain():
    llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
    return ConversationChain(llm=llm)

#uploading a pdf file
st.title("Your document")
file=st.file_uploader("upload a PDF file",type="pdf")
    
## extracting and displaying from pdf
if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)
 
## Breaking data into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
        )
    chunks = text_splitter.split_text(text)

## generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

## creating vector store -FAISS
    vector_store=FAISS.from_texts(chunks,embeddings)
    
    st.session_state.conversation_chain = create_conversation_chain()

    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

## get user question
    user_question=st.text_input("Ask a question about the PDF:")

## do similarity search
    if user_question:
        match=vector_store.similarity_search(user_question)
        #st.write(match)
        
## define the LLM
        llm=ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        
## output results
## chain of events-> take the question,get relevent document,pass it to the LLM,generate output,create session
        if user_question:
            # Retrieve relevant context from the vector store
            context = vector_store.similarity_search(user_question, k=3)  # Retrieve top 3 similar documents
            
            # Prepare the context for the model
            context_text = "\n".join([doc.page_content for doc in context])

            # Append user question to history
            st.session_state.conversation_history.append({"role": "user", "content": user_question})

            # Get the response from the model with context
            response = st.session_state.conversation_chain.predict(
                input=f"{context_text}\n{user_question}",
                history=st.session_state.conversation_history
            )
            
            # Append model response to history
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

            # Display the conversation history
            for message in st.session_state.conversation_history:
                st.write(f"**{message['role'].capitalize()}**: {message['content']}")

# Optional: Clear conversation history
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.conversation_chain = None
