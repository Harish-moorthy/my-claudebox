import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================================
# INTERFACE DESIGN & LAYOUT
# =====================================================================
st.set_page_config(page_title="KNOWLEDGE TERMINAL", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0a0a0a; color: #ffffff; }
        .stTextInput>div>div>input { background-color: #161616; color: white; border: 1px solid #2a2a2a; border-radius: 6px; }
        .stChatInput>div { background-color: #161616 !important; border: 1px solid #2a2a2a !important; }
        div[data-testid="stChatMessage"] { background-color: #161616; border: 1px solid #2a2a2a; border-radius: 8px; margin-bottom: 10px; }
        .stButton>button { background-color: #ffffff; color: #0a0a0a; font-weight: 700; border-radius: 6px; width: 100%; }
        h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, sans-serif; letter-spacing: -0.5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ KNOWLEDGE TERMINAL")
st.caption("Cloud Matrix System // Powered by Gemini 2.5 Flash Engine")

# Fetch the API Key securely from Streamlit Settings Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Initialize session states for storing text documents and chat timelines
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================================
# SIDEBAR: DOCUMENT INGESTION LAYER
# =====================================================================
with st.sidebar:
    st.header("SYSTEM INGESTION")
    uploaded_file = st.file_uploader("Drop knowledge base (PDF format)", type=["pdf"])
    
    if uploaded_file:
        with open("temp_knowledge.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info("File locked into transient memory state.")
        
        if st.button("PROBE & INDEX DOCUMENT"):
            if not GEMINI_API_KEY:
                st.error("Missing API Key. Please add your GEMINI_API_KEY to your Streamlit App Secrets.")
            else:
                with st.spinner("Processing text coordinates..."):
                    # Load and segment the PDF directly into clean text blocks
                    loader = PyPDFLoader("temp_knowledge.pdf")
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
                    final_chunks = text_splitter.split_documents(docs)
                    
                    # Store text blocks into regular memory strings
                    st.session_state.document_chunks = [doc.page_content for doc in final_chunks]
                    st.success("INDEXING COMPLETE. Data node activated.")

# =====================================================================
# MAIN WORKSPACE: CONTEXT-AWARE CONVERSATION LOOPS
# =====================================================================
# Display historical iterations
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about your knowledge files...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Querying knowledge vectors..."):
            if not GEMINI_API_KEY:
                ai_output = "System Lock: Please configure your GEMINI_API_KEY inside the cloud dashboard secrets panel to activate the operational layers."
            elif st.session_state.document_chunks is not None:
                # Scan chunks matching the user query words natively
                query_words = set(user_query.lower().split())
                matched_chunks = []
                
                for chunk in st.session_state.document_chunks:
                    score = sum(1 for word in query_words if word in chunk.lower())
                    if score > 0:
                        matched_chunks.append((score, chunk))
                
                # Sort to pass the best matched context pieces first
                matched_chunks.sort(key=lambda x: x[0], reverse=True)
                context_payload = "\n\n".join([item[1] for item in matched_chunks[:4]])
                
                # Structural Context-Aware System Prompt
                prompt = (
                    f"You are a sophisticated Knowledge Intelligence Agent.\n"
                    f"Analyze the following context pieces thoroughly and answer the user query clearly.\n"
                    f"If the answer cannot be found in the context matrix, state so directly.\n\n"
                    f"Context Matrix:\n{context_payload}\n\n"
                    f"User Query: {user_query}"
                )
                
                # Updated Model String endpoint setup to handle modern production endpoints
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    ai_output = response.json()['contents'][0]['parts'][0]['text']
                else:
                    ai_output = f"API Communication Error: Code {response.status_code}. Server route rejected."
            else:
                # Fallback to direct, standard Gemini intelligence if no document is uploaded yet
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
                payload = {"contents": [{"parts": [{"text": user_query}]}]}
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    ai_output = response.json()['contents'][0]['parts'][0]['text']
                else:
                    ai_output = "Hello! Upload a PDF document in the sidebar, click index, and I will instantly analyze its contents for you."

            st.markdown(ai_output)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_output})
