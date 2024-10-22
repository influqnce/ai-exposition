import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain.schema import Document
import os
import secrets

HF_TOKEN = st.secrets["HF_TOKEN"]


loader = PyPDFLoader("sodapdf-converted.pdf")
#Read the information scraped from all the websites
docs = loader.load()

#split the data
textsplitter = CharacterTextSplitter(separator="\n", chunk_size=1000,chunk_overlap=150,length_function=len)
text_chunks = textsplitter.split_documents(docs)

#LLM Initialization
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

#Vector Database Initialization
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()


#Prompt
template = """

You are a information retrieval AI. Format the retrieved information or text to be 5 sentences long.
Don't mention the document number and pages. 
Do not include any phone numbers or emails. 
Use only the context for your answers, do not make up information and do not make up words.
All questions are in regards to the school 'Sarvodaya Central Vidyalaya'
Answer the question like a person - do not mention that you are an AI.

query: {query}

{context}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")

#Chain Contruction
chain = (
# The initial dictionary uses the retriever and user supplied query
    {"context":retriever,
     "query":RunnablePassthrough()}
# Feeds that context and query into the prompt then model & lastly
# uses the ouput parser, do query for the data.
    |  prompt  | llm | StrOutputParser()

)

# Title of the app
st.title("Sarvodaya Vidyalaya Chatbot")

# Get user input
user_query = st.text_input("Enter your query:")

# When the button is pressed
if st.button("Submit"):
    # Call the function with the user's input
    output = chain.invoke(user_query)
    # Print the output to the screen
    st.write(output)
