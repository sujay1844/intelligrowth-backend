from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import StdOutCallbackHandler
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

from fastapi import FastAPI

data_root = "./data"
loaders=[
	TextLoader(f"{data_root}/RoleandPowerofGovernor.txt"),
	TextLoader(f"{data_root}/Governor’sRoleinUniversities.txt"),
	TextLoader(f"{data_root}/Governor’sPowertodecideonBills-VetoPower.txt"),
	TextLoader(f"{data_root}/ChiefMinister.txt"),
	TextLoader(f"{data_root}/'Union'or'Central'Government.txt"),
	TextLoader(f"{data_root}/InterimReportofJ&KDelimitationCommission.txt"),
	TextLoader(f"{data_root}/Assam-MeghalayaBorderDispute .txt"),
	TextLoader(f"{data_root}/KrishnaWaterDispute.txt"),
	TextLoader(f"{data_root}/StatehoodDemandbyPuducherry.txt"),
	TextLoader(f"{data_root}/BelagaviBorderDispute.txt"),
	TextLoader(f"{data_root}/DemandforIncludingLadakhunderSixthSchedule.txt"),
	TextLoader(f"{data_root}/SpecialCategoryStatus.txt"),
	TextLoader(f"{data_root}/E-ILPPlatform-Manipur.txt"),
	TextLoader(f"{data_root}/LegislativeCouncil.txt"),
	TextLoader(f"{data_root}/GovernmentofNCTofDelhi(Amendment)Act,2021.txt"),
	TextLoader(f"{data_root}/NationalPanchayatiRajDay.txt"),
]
docs=[]
for loader in loaders:
  docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                      chunk_overlap=200,)
#
esops_documents = text_splitter.transform_documents(docs)

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings":True}

embeddings= HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs,
                                  )

persist_docs="chroma"
vector_db=Chroma.from_documents(
    documents=esops_documents,
    embedding=embeddings,
    persist_directory=persist_docs
)

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-64g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)


PROMPT_TEMPLATE = '''
[INST]<<SYS>>
You are my learning assistant.You are very good at creating questions that end with the symbol '?'.
With the information being provided answer the question compulsorly.
If you cant generate a  question based on the information either say you cant  generate .
So try to understand in depth about the context and generate questions only based on the information provided. Dont generate irrelevant questions
<</SYS>>
Context: {context}
Question: {question}
Do provide only helpful answers

Helpful answer:


[/INST]
'''

input_variables = ['context', 'question']

custom_prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                            input_variables=input_variables)


app = FastAPI()

@app.get("/")
def ask(n: int = 5, topics: list = []):
    handler = StdOutCallbackHandler()
    bm25_retriever = BM25Retriever.from_documents(esops_documents)
    bm25_retriever.k=5
    chroma_retriever=vector_db.as_retriever(search_kwargs={"k":5},filter={"source":topics})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,chroma_retriever],weights=[0.5,0.5])
    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        verbose=True,
        callbacks=[handler],
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    topics_list = [topic.replace(".txt", "").replace(f"{data_root}/", "") for topic in topics]
    q_query = f"Give me only {n} questions about {topics_list} which will help me to deepen my understanding and give no answers"
    result = qa_with_sources_chain({'query':q_query})

    a_query = f"Give me only the answers for each of the questions in {result['result']} and dont add anything extra such as \"Of course! I'd be happy to help you with that. Here are five questions\" "
    answers = qa_with_sources_chain({"query":a_query})
    return {
        "questions": result['result'],
        "answers": answers['result'],
    }


