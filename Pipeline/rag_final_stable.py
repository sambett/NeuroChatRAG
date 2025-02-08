import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

load_dotenv()


def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    db = FAISS.load_local("parkinsons_vector_db", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 5})

    template = """You are a Parkinson's disease expert. Follow these rules:
1. Use {language_style} language (technical/simple)
2. Base answers ONLY on these sources
3. Cite sources in your answer
4. If unsure, say "I don't know"

Sources:
{context}

Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    def build_context(inp):
        if isinstance(inp.get("question"), dict):
            question_text = inp["question"].get("question", "")
        else:
            question_text = inp.get("question", "")

        if isinstance(inp.get("language_style"), dict):
            language_style_text = inp["language_style"].get("language_style", "")
        else:
            language_style_text = inp.get("language_style", "")

        docs = retriever.get_relevant_documents(question_text)
        context = " ".join(doc.page_content for doc in docs)
        print(f"Retrieved context for question '{question_text}':\n{context}\n")

        return {
            "question": question_text,
            "language_style": language_style_text,
            "context": context
        }

    rag_chain = (
        RunnableMap({
            "question": RunnablePassthrough(),
            "language_style": RunnablePassthrough()
        })
        | build_context
        | prompt
        | llm
    )

    return rag_chain

def test_rag_pipeline():
    rag_chain = load_rag_pipeline()

    test_questions = [
        {
            "question": "What are the early symptoms of Parkinson's disease?",
            "language_style": "simple"
        },
        {
            "question": "How does autophagy dysfunction contribute to the progression of Parkinson's disease?",
            "language_style": "technical"
        },
        {
            "question": "What are the main genetic mutations associated with Parkinson's disease?",
            "language_style": "technical"
        },
        {
            "question": "What role does neuroinflammation play in Parkinson's disease?",
            "language_style": "simple"
        },
        {
            "question": "How do exosomes contribute to intercellular communication in Parkinson's disease?",
            "language_style": "technical"
        },
        {
            "question": "What are the current diagnostic techniques for Parkinson's disease?",
            "language_style": "simple"
        }
    ]

    for test in test_questions:
        response = rag_chain.invoke(test)
        print(f"Question: {test['question']}")
        print(f"Language Style: {test['language_style'].upper()}")
        print(f"Answer: {response.content}\n{'='*50}\n")

if __name__ == "__main__":
    test_rag_pipeline()