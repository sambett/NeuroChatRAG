from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv  

load_dotenv()

def load_rag_pipeline():
    # 1. Load embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    db = FAISS.load_local("parkinsons_vector_db", embeddings, allow_dangerous_deserialization=True)

    # 2. Set up retriever with an increased number of documents to retrieve (k=5)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3. Define prompt template
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

    # 4. Load LLM (the API key is set via the environment variable)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 5. Define a function to build and log the context
    def build_context(inp):
        # Unwrap "question" if it is a dict.
        question = inp["question"] if isinstance(inp["question"], str) else inp["question"].get("value", "")
        # Retrieve relevant documents from the FAISS index.
        docs = retriever.get_relevant_documents(question)
        # Build the context string by joining the content of each document.
        context = " ".join(doc.page_content for doc in docs)
        # Log the retrieved context.
        print("Retrieved context for question '{}':\n{}\n".format(question, context))
        # Return the updated input dictionary.
        return {**inp, "question": question, "context": context}

    # 6. Create the RAG chain by chaining the steps.
    rag_chain = (
        RunnableMap({
            "question": RunnablePassthrough(),
            "language_style": RunnablePassthrough()
        })
        | build_context   # This step logs the retrieved context.
        | prompt
        | llm
    )

    return rag_chain

def test_rag_pipeline():
    # Load the RAG chain
    rag_chain = load_rag_pipeline()

    # Test cases with additional questions to probe for different sections:
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

    # Run tests and print the output for each test.
    for test in test_questions:
        response = rag_chain.invoke(test)
        print(f"Question: {test['question']}")
        print(f"Language Style: {test['language_style'].upper()}")
        print(f"Answer: {response.content}\n{'='*50}\n")

if __name__ == "__main__":
    test_rag_pipeline()
