from graph import app
from langchain_core.messages import HumanMessage

def main():
    print("==========================================")
    print("🤖 MULTIMODAL RAG AGENT (GROQ EDITION)")
    print("Drop PDFs, Images, Audio in /data")
    print("Run `persist_ingest.py` or `ingest.py` first!")
    print("==========================================")

    chat_history = []
    
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Append User Message to History
        chat_history.append(HumanMessage(content=user_input))
        
        # Run the Agent Graph with History
        initial_state = {"messages": chat_history, "next_step": ""}
        
        result = app.invoke(initial_state)
        
        # Get Final Response (Last AI Message)
        # The graph returns the updated state. We want to grab the NEW messages.
        # But efficiently, we can just grab the last message content.
        final_msg_obj = result["messages"][-1]
        final_msg = final_msg_obj.content
        
        # Update History with valid messages only (avoid duplicates if graph returns all)
        # LangGraph usually returns the final state.
        chat_history = result["messages"]
        
        print(f"\n🤖 AI: {final_msg}")
        
        # Check if it returned Mermaid Code (Visualization)
        if "graph TD" in final_msg or "graph LR" in final_msg:
            print("\n[INFO] Flowchart detected! Copy the code above into https://mermaid.live to view it.")

if __name__ == "__main__":
    main()