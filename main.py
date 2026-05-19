

import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

print("All Imports Working")

app = FastAPI()


os.environ["GROQ_API_KEY"] = "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"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever()




llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question only using the provided context.

    Context:
    {context}

    Question:
    {question}
    """
)



class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):

    query = request.question


    retrieved_docs = retrieved_docs = retriever.invoke(query)

    
    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    formatted_prompt = prompt.format(
        context=context,
        question=query
    )

    
    response = llm.invoke(formatted_prompt)

    return {
        "question": query,
        "answer": response.content
    }
