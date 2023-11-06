from transformers import pipeline

from model.model_loader import model, tokenizer
from prompts.feedback_prompt import feedback_prompt

feedback_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

def get_feedback(question, response, expected):
    prompt = feedback_prompt.format(question=question, response=response, expected=expected)

    return feedback_generator(prompt)[0]['generated_text']
