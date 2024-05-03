import os
import boto3

import json

import sys
import faiss

import streamlit as st

###################################################################################################
## we will be using TITAN EMBEDDING MODELS to generate embedding as follows:                      #

from langchain_community.embeddings import BedrockEmbeddings

from langchain.llms.bedrock import Bedrock

###################################################################################################
## 1. Data Ingestion                                                                              #

import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFDirectoryLoader

###################################################################################################
## 2. Vector Embeddings and Vector Store                                                          #

from langchain_community.vectorstores import FAISS

###################################################################################################
## 3. LLM Models                                                                                  #

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

###################################################################################################
## 4. BEDROCK Clients                                                                             #

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                     client=bedrock
                                     )

###################################################################################################
## 5. Data Ingestion                                                                              #

def data_ingestion():
    loader=PyPDFDirectoryLoader("Data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)

    return docs

###################################################################################################
## 6. Vector Embeddings                                                                           #

def get_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )

    vector_store_faiss.save_local("faiss_index")


def get_llama2_llm():
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",
                client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

Prompt_template="""
Human: Use the following pieces of context to provide a concise answer to the question at the end but aleast
summarize with 250 words with detailed explanations.

If you don't know the answer, just say that you don't know the anser. Don't try to make up the answer.
<context>
{context}
</context>

Question  : {question}

Assistant : """

PROMPT = PromptTemplate(template=Prompt_template, input_variables=["context","question"])

def get_response_llm(llm,vector_store_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store_faiss.as_retriever(search_type="similarity", search_kwargs={"k":3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
        )   
    
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock!")

    user_question=st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing...."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)

            llm = get_llama2_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ =="__main__":
    main()