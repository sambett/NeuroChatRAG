import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

# Load environment variables (e.g., for API keys)
load_dotenv()

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    # Initialize embeddings and load your FAISS vector database.
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")
    db = FAISS.load_local("parkinsons_vector_db", embeddings, allow_dangerous_deserialization=True)

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
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the language model.
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Function to build the context from retrieved documents.
    def build_context(inp):
        # Extract question and language style from the input.
        if isinstance(inp.get("question"), dict):
            question_text = inp["question"].get("question", "")
        else:
            question_text = inp.get("question", "")

        if isinstance(inp.get("language_style"), dict):
            language_style_text = inp["language_style"].get("language_style", "")
        else:
            language_style_text = inp.get("language_style", "")

        # Retrieve relevant documents.
        docs = retriever.get_relevant_documents(question_text)
        context = " ".join(doc.page_content for doc in docs)
        st.write(f"Retrieved context for question '{question_text}':\n{context}\n")

        return {
            "question": question_text,
            "language_style": language_style_text,
            "context": context
        }

    # Build the RAG pipeline.
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

def main():
    st.title("Parkinson's Disease Q&A")
    st.write("Ask questions about Parkinson's disease and get answers based on trusted sources.")

    # Load the RAG pipeline (this is cached to avoid reloading on every run).
    rag_chain = load_rag_pipeline()

    # Input fields for the question and language style.
    question_input = st.text_input("Enter your question:")
    language_style = st.selectbox("Select language style", ["simple", "technical"])

    if st.button("Get Answer"):
        if not question_input.strip():
            st.error("Please enter a question.")
        else:
            # Prepare the input dictionary for the RAG pipeline.
            input_data = {
                "question": question_input,
                "language_style": language_style
            }
            with st.spinner("Generating answer..."):
                # Get the response from the RAG pipeline.
                response = rag_chain.invoke(input_data)
            st.subheader("Answer:")
            st.write(response.content)

if __name__ == "__main__":
    main()
