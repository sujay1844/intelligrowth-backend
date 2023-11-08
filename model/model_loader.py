from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model_name = "BAAI/bge-small-en-v1.5"
model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"

embeddings= HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings":True},
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
