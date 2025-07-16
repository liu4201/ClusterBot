#============================================================
# ClusterBot (cb) version 1.0
# Author: Xiao Liu
# liu4201@purdue.edu
# Date: 2024-11-01
#------------------------------------------------------------

import os
import sys
import json
import tracemalloc
import logging
from typing import List, Tuple, Dict
import shutil
import argparse
import hashlib

from transformers import pipeline
import torch
import requests
import time
from datetime import datetime
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
_ = load_dotenv(find_dotenv()) # read local .env file

api_key  = os.environ['AI_API_KEY']

#sys.path.append('../../ClusterBot') 

# for flask
from flask import Flask, render_template, request, redirect, Response, jsonify, session


def get_answer(question):
    docs = vectordb.similarity_search(question,k=3)

    url = "https://genai.rcac.purdue.edu/ollama/api/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model_name,
        "messages": [
        {
            "role": "system",
            "content": "You are a helper that take knowledge below and advice: " + docs[0].page_content + docs[1].page_content+ docs[2].page_content
        },
        {
            "role": "user",
            "content": question   
        }
        ],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=body)
    print(response)

    return docs, json.loads(response.text)["message"]["content"]

def get_answer_from_local(question):
    docs = vectordb.similarity_search(question,k=3)

    messages = [
        {"role": "system", "content": "You are a helper that take knowledge below and advice: " + docs[0].page_content + docs[1].page_content+ docs[2].page_content},
        {"role": "user", "content": question},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=512,
    )

    return docs, outputs[0]["generated_text"][-1]["content"]

class DocumentManager:
    def __init__(self, directory_path, glob_pattern="./*.md"):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.documents = []
        #self.sections = []
        self.texts = []
    
    def load_documents(self, split_on_header=False):
        if split_on_header:
            loader = NotionDirectoryLoader(self.directory_path)
        else:
            loader = DirectoryLoader(self.directory_path, glob=self.glob_pattern, show_progress=True, loader_cls=UnstructuredMarkdownLoader)
        self.documents = loader.load()

    def split_documents(self, split_on_header=False):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #c_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0.25, separator="\n\n")
        r_splitter = RecursiveCharacterTextSplitter(chunk_size=split_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n"])
        for doc in self.documents:
            sections = md_header_splitter.split_text(doc.page_content)
            source = doc.metadata["source"].split("/")[-1]

            if split_on_header:

                texts = r_splitter.split_documents(sections)
    
                for txt in texts:
                    txt.metadata["source"] = source
        
                self.texts.extend(texts)

            else:

                for section in sections:
                    texts = r_splitter.split_text(section.page_content)
                    texts = [Document(page_content=text, metadata={"source": source}) for text in texts]
                    self.texts.extend(texts)

######################## flask ########################
app = Flask(__name__, static_url_path='/static')

# Configure logging
log_dir = 'logs'  # Optional: Create a directory for logs
os.makedirs(log_dir, exist_ok=True)

date_str = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(log_dir, f'app_{date_str}.log')
log_level = logging.INFO

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(log_level)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

#logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
#logger = logging.getLogger()
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)





@app.route('/')
def output():
    # serve index template
    return render_template('index.html')

@app.route("/send_question", methods=["POST"])
def chat():

    data = request.json
    question = data.get("question")
    #print(question)
    start_time = time.perf_counter()

    if args.local:
        source, answer = get_answer_from_local(question)
    else:
        source, answer = get_answer(question)

    end_time = time.perf_counter()

    out_dict = {"answer":answer, 
        "source0": source[0].metadata["source"],"cite0":source[0].page_content,
        "source1": source[1].metadata["source"],"cite1":source[1].page_content,
        "source2": source[2].metadata["source"],"cite2":source[2].page_content}

    
    elapsed_time = end_time - start_time
    #print("====================source====================")
    #print(source)
    #print("====================answer====================")
    #print(out_dict)
    #print(answer)
    print("====================time====================")
    print(f"Elapsed time (perf_counter): {elapsed_time:.2f} seconds")
    print("====================time====================")

    app.logger.info(f"question: {question}, {out_dict}")


    return jsonify(out_dict) #jsonify({"sucess": question})# 

@app.route('/protected')
def protected_route():
    if session.get('verified'):
        return render_template('index.html')
    else:
        return redirect('/')


######################## flask ########################


if __name__ == "__main__":

    # step 0: set arguments
    parser = argparse.ArgumentParser(description="Description")
    
    #Optional arguments
    parser.add_argument("-s", "--split_size", type=int, default=1500, help="the chunk size to split document")
    parser.add_argument("--chunk_overlap", type=float, default=0.25, help="the overlapped ratio between two chunks")
    parser.add_argument("-m", "--model", type=str, default="llama3.3:70b", help="the LLMs")
    parser.add_argument("-e", "--embedding", type=str, default="all-mpnet-base-v2", help="the embedding models")
    parser.add_argument('--local', action='store_true', help='Enable my local LLMs: llama3.2-Instruct')
    parser.add_argument('--split_on_header', action='store_true', help='Enable split on header with SupportKnowledgeBase')


    args = parser.parse_args()

    split_size = args.split_size
    chunk_overlap = args.chunk_overlap
    model_name = args.model
    embedding_name = args.embedding

    # step 1: read in LLM
    if args.local:
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )


    #step 2: read in vector database
    doc_manager = DocumentManager('/Users/xiaoliu/Work/SupportKnowledgeBase.wiki')
    doc_manager.load_documents(split_on_header=args.split_on_header)
    doc_manager.split_documents(split_on_header=args.split_on_header)

    embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/" + embedding_name)
    #embedding_model = HuggingFaceEmbeddings(model_name= "/Users/xiaoliu/Work/Project/ClusterBot/hf_model/all-mpnet-base-v2" )

    #embedding_model = HuggingFaceEmbeddings(model_name="meta-llama/Llama-3.2-3B")
    #embedding_model.client.tokenizer.pad_token = embedding_model.client.tokenizer.eos_token
    #embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/" + embedding_name)

    persist_directory = '/Users/xiaoliu/Work/Project/ClusterBot/doc/chroma'

    try:
        shutil.rmtree(persist_directory)
        print(f"Folder '{persist_directory}' deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{persist_directory}' not found.")
    except OSError as e:
        print(f"Error deleting folder '{persist_directory}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    vectordb = Chroma.from_documents(documents=doc_manager.texts, embedding=embedding_model, persist_directory=persist_directory)
    print(vectordb._collection.count())

    print(" model & data Loaded!")

    # step 3: start the flask server
    app.run(port=8000, debug=False)

