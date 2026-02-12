from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import TEXTBOOKS_DB_DIR, EMBEDDING_MODEL, RAG_TOP_K

def retrieve_context(query: str, db_path: str = str(TEXTBOOKS_DB_DIR), top_k: int = RAG_TOP_K):
    """Phase 3.2: Retrieval with textbook context"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    try:
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        results = vector_db.similarity_search(query, k=top_k)
        if not results:
             return "No relevant context found."
        return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in results])
    except Exception as e:
        return f"Retrieval failed: {e}. Is the vector database populated?"

def hyde_retrieval_logic(query: str, llm=None):
    """
    Phase 3.2 HyDE implementation.
    Generates a hypothetical document using the LLM and then uses that to retrieve context.
    """
    if llm is None:
        # Fallback if no LLM provided (e.g. testing)
        print("Warning: No LLM provided for HyDE. Falling back to standard retrieval.")
        return retrieve_context(query)
    
    hyde_prompt = ChatPromptTemplate.from_template(
        "You are a physics expert. Write a short, textbook-quality explanation specifically answering this question: {query}"
    )
    chain = hyde_prompt | llm | StrOutputParser()
    try:
        hypo_doc = chain.invoke({"query": query})
        print(f"Generated Hypothetical Document (first 50 chars): {hypo_doc[:50]}...")
        return retrieve_context(hypo_doc)
    except Exception as e:
         print(f"HyDE generation failed: {e}. Falling back to standard retrieval.")
         return retrieve_context(query)

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Newton's second law of motion"
    print(f"Query: {query}")
    print("-" * 30)
    context = retrieve_context(query)
    print(context)
