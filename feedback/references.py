from database.chroma import vector_db

def get_references(question, expected):
    qna = question + "\n" + expected

    return vector_db.similarity_search(qna, k=1)[0].page_content