import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv  

load_dotenv()

# Configure logging: set level to DEBUG for detailed output.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_rag_pipeline():
    logging.info("Loading embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    db = FAISS.load_local("parkinsons_vector_db", embeddings, allow_dangerous_deserialization=True)

    logging.info("Setting up retriever with k=5...")
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Define the prompt template.
    template = """You are a Parkinson's disease expert. Follow these rules:
1. Use {language_style} language (technical/simple)
2. Base answers ONLY on these sources
3. Cite sources in your answer
4. If unsure, say "I don't know"

Sources:
{context}

Question: {question}
Answer:"""
    logging.info("Creating prompt template...")
    prompt = ChatPromptTemplate.from_template(template)

    logging.info("Loading ChatOpenAI model (gpt-3.5-turbo)...")
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Define a function to build and log the context.
    def build_context(inp):
        logging.debug("build_context received input: %s", inp)
        # Extract the question text robustly.
        if isinstance(inp.get("question"), dict):
            # Check for key 'question' inside the nested dict.
            question_text = inp["question"].get("question", "")
        else:
            question_text = inp.get("question", "")
        # Similarly, extract the language_style.
        if isinstance(inp.get("language_style"), dict):
            language_style_text = inp["language_style"].get("language_style", "")
        else:
            language_style_text = inp.get("language_style", "")

        logging.info("Retrieving documents for question: '%s'", question_text)
        docs = retriever.get_relevant_documents(question_text)
        context = " ".join(doc.page_content for doc in docs)
        logging.info("Retrieved %d documents for question '%s'.", len(docs), question_text)
        logging.debug("Retrieved context: %s", context)

        # Return a flattened dictionary for the subsequent steps.
        return {
            "question": question_text,
            "language_style": language_style_text,
            "context": context
        }

    # Create the RAG chain by chaining the steps.
    logging.info("Creating the RAG pipeline chain...")
    rag_chain = (
        RunnableMap({
            "question": RunnablePassthrough(),
            "language_style": RunnablePassthrough()
        })
        | build_context  # Build and log the context.
        | prompt
        | llm
    )

    logging.info("RAG pipeline loaded successfully.")
    return rag_chain

def test_rag_pipeline():
    logging.info("Starting RAG pipeline tests...")
    rag_chain = load_rag_pipeline()

    # Define several test cases probing different aspects of Parkinson's disease.
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

    # Process each test question through the chain and print the results.
    for test in test_questions:
        logging.info("Processing test question: '%s'", test["question"])
        response = rag_chain.invoke(test)
        logging.info("Completed processing question: '%s'", test["question"])
        print(f"Question: {test['question']}")
        print(f"Language Style: {test['language_style'].upper()}")
        print(f"Answer: {response.content}\n{'='*50}\n")

if __name__ == "__main__":
    test_rag_pipeline()
