import os
from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from tools import search_knowledge_base

# 1. Force Load Environment Variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ CRITICAL ERROR: GROQ_API_KEY is missing from .env file!")

# 2. Define State
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next_step: str

# 3. Initialize LLM (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0, 
    api_key=api_key 
)

# 4. Define Nodes
def supervisor_node(state: AgentState):
    messages = state['messages']
    last_user_msg = messages[-1].content.lower()
    
    if "visualize" in last_user_msg or "flowchart" in last_user_msg:
        return {"next_step": "VISUALIZER"}
    elif "don't understand" in last_user_msg:
        return {"next_step": "VISUALIZER"}
    
    # If we have an AI message (from Researcher), we are done
    if isinstance(messages[-1], AIMessage):
        return {"next_step": "FINISH"}

    return {"next_step": "RESEARCHER"}

def researcher_node(state: AgentState):
    last_message = state['messages'][-1].content
    print(f"🕵️ Researcher is looking up: {last_message}")
    context = search_knowledge_base(last_message) # Renamed 'result' to 'context' for clarity with new prompt

    # Assuming llm_provider is set globally or passed, for this example, we'll assume it's "Groq"
    # and that the global 'llm' object is the one to use for synthesis.
    # If 'llm_provider' is not defined, this block will not execute.
    llm_provider = "Groq" # Placeholder, adjust as needed based on actual setup

    if llm and llm_provider != "Mock":
        print(f"Synthesizing answer using {llm_provider}...")
        
        # Enhanced Prompt for Synthesis (Walkthrough aligned)
        prompt = (
            f"You are an intelligent expert. User asks: '{last_message}'\n\n"
            f"Here is the retrieved context from multiple sources (PDFs, Audio, Images):\n"
            f"---------------------\n{context}\n---------------------\n\n"
            f"Check the User's Intent and Apply the Correct Mode:\n"
            f"1. **Compare Mode**: If user asks to 'Compare' A vs B, explicitly list Differences and Similarities.\n"
            f"2. **Formula Mode**: If identifying formulas, list them as '[Doc Name]: Formula'.\n"
            f"3. **Adaptive Mode**: \n"
            f"   - If user says 'Explain in detail', provide a multi-paragraph deep dive.\n"
            f"   - If user says 'Summarize', provide bullet points.\n"
            f"4. **Concise Mode**: For general queries, be brief and focused. Read everything but report only key takeaways.\n"
            f"5. **Holistic Mode**: Synthesize facts into ONE narrative. Avoid repetitive lists.\n"
            f"6. **Attribution Mode**: If user asks 'Who wrote...', 'Which author...', or 'Find source...', analyze content AND filenames to identify the creator involved with the specific concept (e.g., 'Love').\n\n"
            f"Answer:"
        )

        try:
            # Handle different invoke patterns if necessary, but LangChain/Groq/Ollama usually support invoke or similar
            if llm_provider == "Groq":
                 # Using the global 'llm' which is ChatGroq
                 res = llm.invoke(prompt) # ChatGroq's invoke method
                 content = res.content
            # The original snippet had specific Groq/Ollama client calls,
            # but since 'llm' is already a LangChain ChatGroq, we use its invoke.
            # For other providers, you'd need to initialize 'llm' differently or add more logic.
            elif llm_provider == "Ollama":
                # This would require 'llm' to be an Ollama client or a LangChain Ollama model
                # For now, assuming 'llm' is ChatGroq, this path won't be taken.
                # If 'llm' was an Ollama client, it might look like:
                # res = llm.generate(model='mistral', prompt=prompt, stream=False)
                # content = res['response'] if isinstance(res, dict) else str(res)
                res = llm.invoke(prompt) # Fallback to generic invoke if 'llm' is a LangChain model
                content = res.content if hasattr(res, 'content') else str(res)
            else: # OpenAI or LangChain wrapper
                res = llm.invoke(prompt)
                content = res.content if hasattr(res, 'content') else str(res)
                
            # Pass Context + Draft to Reviewer
            combined_content = f"CONTEXT_BLOCK:\n{context}\n\n---DRAFT_BLOCK---\n{content}"
            response_msg = AIMessage(content=combined_content)
        except Exception as e:
            print(f"LLM Synthesis failed: {e}")
            if "429" in str(e):
                error_msg = "⚠️ API Rate Limit Hit (Groq Free Tier). Please wait a moment and try again."
            else:
                error_msg = f"(LLM Synthesis Failed: {e})"
            
            response_msg = AIMessage(content=f"Found Facts: {context}\n\n{error_msg}")
    else:
        response_msg = AIMessage(content=f"Found Facts: {context}")
        
    return {"messages": [response_msg]}

def reviewer_node(state: AgentState):
    messages = state['messages']
    user_query = messages[-2].content if len(messages) > 1 else "Unknown"
    # Parse Context vs Draft
    content = messages[-1].content
    if "CONTEXT_BLOCK:" in content and "---DRAFT_BLOCK---" in content:
        try:
            parts = content.split("---DRAFT_BLOCK---")
            context_part = parts[0].replace("CONTEXT_BLOCK:", "").strip()
            draft_part = parts[1].strip()
        except:
            context_part = "UNKNOWN"
            draft_part = content
    else:
        context_part = "UNKNOWN"
        draft_part = content

    print("🧐 Reviewer is critiquing the draft...")

    prompt = f"""
    You are a Senior Editor and Fact-Checker.
    
    User Query: {user_query}
    
    Original Retrieved Context (Truth):
    {context_part}
    
    Researcher's Draft Answer:
    {draft_part}
    
    Task:
    1. Verify that the Draft is fully supported by the Context. 
    2. **HALLUCINATION CHECK**: If the draft contains claims NOT in the context, remove them or correct them.
    3. If the Context is empty or irrelevant, admitting "I don't know" is better than hallucinating.
    3. If the Context is empty or irrelevant, admitting "I don't know" is better than hallucinating.
    4. Refine the tone to be professional and clear.
    5. **ATTRIBUTION CHECK**: Ensure authors/sources are correctly linked to their concepts/poems.
    
    Output the Final Polished Answer:
    """
    
    if llm:
        try:
            response = llm.invoke(prompt)
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as e:
             print(f"Reviewer failed: {e}")
             if "429" in str(e):
                 # If reviewer hits 429, just return the Draft with a warning note
                 return {"messages": [AIMessage(content=draft_part + "\n\n(Review skipped due to Rate Limit 429)")]}
             return {"messages": [AIMessage(content=draft_part)]} # Fallback to draft
    else:
        return {"messages": [AIMessage(content=draft_part)]}

def visualizer_node(state: AgentState):
    context = state['messages'][-2].content if len(state['messages']) > 1 else state['messages'][-1].content
    print("🎨 Visualizer is drawing a flowchart...")
    prompt = f"""
    Based on this context: {context}
    
    User Query: {state['messages'][-1].content}
    
    Task:
    1. If the user said "I don't understand", first provide a **Key Points** summary (bullet points) to clarify the concept.
    2. Then, create a MERMAID.JS flowchart to visualize it.
    
    Output Format:
    [Key Points explanation if needed]
    
    ```mermaid
    graph TD
    ...code...
    ```
    """
    try:
        response = llm.invoke(prompt)
        # We return the full content now (Text + Code) so the user sees the clarification.
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        print(f"Visualizer failed: {e}")
        if "429" in str(e):
             return {"messages": [AIMessage(content="(Visualization skipped due to API Rate Limit. Please try again later.)")]}
        return {"messages": [AIMessage(content="(Visualization Failed)")]}


# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("visualizer", visualizer_node)
workflow.set_entry_point("supervisor")

def route_logic(state):
    return state['next_step']

workflow.add_conditional_edges("supervisor", route_logic, {
    "RESEARCHER": "researcher",
    "VISUALIZER": "visualizer",
    "FINISH": END
})
workflow.add_edge("researcher", "reviewer")
workflow.add_edge("reviewer", "supervisor")
workflow.add_edge("visualizer", END)
app = workflow.compile()