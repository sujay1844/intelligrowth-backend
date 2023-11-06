from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import re

from qa.qa_gen import QAGenerator
from feedback.references import get_references
from feedback.feedback_gen import get_feedback
from keywords.keywords_gen import get_missing_keywords
from api.models import QAApiBody, FeedbackApiBody

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=["*"],
)

questions = []
answers = []

@app.post("/qa")
def ask(apiBody: QAApiBody):
    qa_chain = QAGenerator(apiBody.n, apiBody.topics)

    global questions, answers
    questions = qa_chain.get_questions()
    answers = qa_chain.get_answers()

    return {
        "questions": questions,
        "answers": answers,
    }

@app.post("/q")
def get_question(n: int):
    global questions, answers

    return {
        "question": questions[n],
        "answer": answers[n],
    }

@app.post("/feedback")
def generate_keywords(apiBody: FeedbackApiBody):

    question = apiBody.question
    response = apiBody.response
    expected = apiBody.expected

    reference = get_references(question, expected)

    feedback = get_feedback(question, response, expected)
    feedback = re.sub(r'[INST].*[/INST]', '', feedback)

    return {
        "missing_keywords": get_missing_keywords(response,expected),
        "feedback": feedback,
        "references": reference,
    }

@app.get("/clear")
def clear():
    global questions, answers
    questions = []
    answers = []
    return "Cleared"

@app.get("/ping")
def ping():
    return "pong"