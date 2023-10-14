import langchain
from langchain.embeddings import CacheBackedEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.llms import HuggingFacePipeline
from langchain.cache import InMemoryCache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import prompt
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain import PromptTemplate
from langchain.globals import set_llm_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from keybert.llm import TextGeneration
from keybert import KeyLLM, KeyBERT

from keywords_prompt import keyword_prompt

store = LocalFileStore("./cache")

dir_loader = DirectoryLoader(
    "/kaggle/input/cbse-old-times",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
docs = dir_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

split_documents = text_splitter.transform_documents(docs)

store = LocalFileStore("./cache/")
embed_model_id = 'BAAI/bge-small-en-v1.5'
core_embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id)
embedder = CacheBackedEmbeddings.from_bytes_store(
    core_embeddings_model,
    store,
    namespace=embed_model_id
)
# Create VectorStore
vectorstore = FAISS.from_documents(split_documents,embedder)

PROMPT_TEMPLATE="""Act as a strict professor who has immense knowledge about the given document.
Generate a list of questions that cover the following categories : 
Factual Questions : Answered with specific facts from the document. 
Conceptual Questions : require the reader to understand the main concepts in the document. 
Critical thinking Questions : require the reader to analyze and evaluate the information in the document. 
Creative questions : ask the reader to come up with new ideas or perspectives based on the information in the document.
Every question must align with the main points of the document. 
Please strictly generate 5 questions along with detailed answers based on the following document about [Insert Document Topic]. Ensure that the questions are relevant and the answers are accurate, using information from the document.
Context: {context}
Question: {question}
Do provide only helpful answers

Helpful answer:
"""

input_variables=['context','question']

custom_prompt=PromptTemplate(template=PROMPT_TEMPLATE,input_variables=input_variables)

bm25_retriever = BM25Retriever.from_documents(split_documents)
bm25_retriever.k=5

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=False,
    revision="gptq-8bit-32g-actorder_True"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k":5})
bm25_retriever = BM25Retriever.from_documents(split_documents)
bm25_retriever.k=5

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever,faiss_retriever],
    weights=[0.5,0.5]
)


set_llm_cache(InMemoryCache())

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = ensemble_retriever,
    callbacks=[handler],
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

keyword_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    repetition_penalty=1.1
)
llm = TextGeneration(keyword_pipe, prompt=keyword_prompt)
kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

from flask import Flask, jsonify, request
import requests_cache
requests_cache.install_cache('api_cache', backend='sqlite', expire_after=180)


app = Flask(__name__)

@app.route('/query')
def query():
    data = request.get_json()
    chapter = data.get('chapter')
    query = f"Give me questions on the chapter number {chapter}"
    response = qa_with_sources_chain({"query":query})
    return jsonify({
        "response": response,
    })

@app.route('/keywords')
def get_keywords():
    data = request.get_json()
    doc = data.get('doc')
    keywords = kw_model.extract_keywords([doc], threshold=.5)
    return jsonify({
        "keywords": keywords,
    })

