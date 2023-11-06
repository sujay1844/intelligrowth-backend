from transformers import pipeline
from keybert.llm import TextGeneration
from keybert import KeyLLM

from model.model_loader import model, tokenizer
from prompts.keyword_prompt import keyword_prompt

keyword_generator = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)

key_llm = TextGeneration(keyword_generator, prompt=keyword_prompt)
kw_model = KeyLLM(key_llm)

def get_missing_keywords(response, expected):
    response_keywords = kw_model.extract_keywords(response)[0]
    expected_keywords = kw_model.extract_keywords(expected)[0]

    return list(set(expected_keywords) - set(response_keywords))