import streamlit as st
import os
import nest_asyncio
nest_asyncio.apply()
from langchain_core.messages import HumanMessage, AIMessage
from graph import app

# --- Page Config ---
st.set_page_config(
    page_title="Multimodal RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Header ---
st.title("🤖 Multimodal RAG Agent")
st.markdown("Query your PDFs, Images, Audio, and PPTs using an intelligent multi-agent system.")

# --- Session State for History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom CSS for Chat Interface ---
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Display History ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# --- Input Box ---
user_input = st.chat_input("Ask about your documents...")

if user_input:
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 2. Run Agent Graph
    initial_state = {"messages": st.session_state.messages, "next_step": ""}
    
    with st.spinner("🤖 Agent is thinking... (Researching & Reviewing)"):
        try:
            result = app.invoke(initial_state)
            
            # 3. Get Final Response
            final_msg_obj = result["messages"][-1]
            final_content = final_msg_obj.content
            
            # 4. Display AI Response
            with st.chat_message("assistant"):
                st.markdown(final_content)
                
                # Check for Mermaid Code
                if "graph TD" in final_content or "graph LR" in final_content:
                    st.info("🎨 Flowchart detected above. Rendering...")
                    # Basic Markdown rendering (Streamlit supports mermaid in st.markdown as of recent updates usually, 
                    # but if not, user sees code block which is fine for 'simple')
            
            # 5. Update History
            # We append the result to session state carefully to avoid duplicates if langgraph returns full history
            # But simpler is just to append the final AIMessage
            st.session_state.messages.append(final_msg_obj)
            
        except Exception as e:
            st.error(f"❌ Error during execution: {e}")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Data Status")
    if os.path.exists("./db"):
        st.success("Vector Database Found")
    else:
        st.warning("Vector Database Missing!")
        st.info("Please run `ingest.py` first.")
    
    st.header("⚙️ Configuration")
    st.code(f"Model: Llama 3.3 (Groq)\nEmbeddings: Local (BGE)\nAgents: Researcher, Reviewer, Visualizer", language="text")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
