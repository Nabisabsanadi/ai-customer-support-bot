import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="AI Customer Support Bot", page_icon="🤖")

@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    product_documents = [
        "iPhone 14: Features a 6.1-inch Super Retina XDR display, dual-camera system, A15 Bionic chip with 5-core GPU, and all-day battery life. Price: ₹69,900.",
        "ProWireless Earbuds: Active Noise Cancellation, Bluetooth 5.3, 30-hour total battery with charging case, IPX4 water resistant. Price: ₹2,999.",
        "FitPulse Smartwatch: 1.4-inch AMOLED display, heart rate & SpO2 monitor, 7-day battery life, 50+ sports modes. Price: ₹4,499.",
        "Shipping & Return Policy: Standard delivery takes 3-5 business days. Free shipping on orders above ₹999. Easy 7-day return policy."
    ]
    vector_store = FAISS.from_texts(product_documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256, temperature=0.3)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    template = """
    You are a helpful and professional AI Customer Support Agent.
    Use the retrieved context below to answer the customer's query accurately.

    Retrieved Context: {context}
    Customer Name: {customer_name}
    Customer Query: {query}

    Support Response:
    """
    prompt = PromptTemplate(input_variables=["context", "customer_name", "query"], template=template)
    support_chain = prompt | llm
    return retriever, support_chain

retriever, support_chain = load_rag_system()

st.title("⚡ AI Customer Support Chatbot")
st.write("Powered by LangChain, FAISS, and Hugging Face FLAN-T5 (RAG).")

customer_name = st.text_input("Your Name", "Rahul")
query = st.text_input("Ask your query (e.g., iPhone price, return policy)")

if st.button("Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context_text = docs[0].page_content if docs else "Standard guidelines apply."
            response = support_chain.invoke({"context": context_text, "customer_name": customer_name, "query": query})
            st.success(response.strip())
    else:
        st.warning("Pehle koi query toh daalo!")
