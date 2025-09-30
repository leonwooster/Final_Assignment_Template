"""LangGraph state graph construction for the LLM-powered agent."""

import base64
import json
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from .constants import CACHE_DIR
from .tools import tool_classes


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Global variables that will be initialized in build_agent_graph()
_tools = None
_agent_chain = None
_generation_chain = None

def _initialize_chains_and_tools():
    """Initialize the tools and LLM chains. Called once when building the graph."""
    global _tools, _agent_chain, _generation_chain
    
    if _tools is not None:
        return  # Already initialized
    
    # Initialize the LLM
    # Using gpt-4o which has excellent tool-calling support
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        verbose=True  # Show LLM calls
    )
    
    # Instantiate the tools
    _tools = []
    for tool in tool_classes:
        if callable(tool) and not isinstance(tool, type):
            # It's a function that returns a tool instance (like create_wikipedia_tool)
            _tools.append(tool())
        else:
            # It's a class, instantiate it
            _tools.append(tool())
    
    # CRITICAL: Bind tools to the LLM using OpenAI's native function calling
    llm = llm.bind_tools(_tools)
    
    # Render the tools to a text description for the prompt
    rendered_tools = render_text_description(_tools)
    
    # Create the system prompt
    system_prompt = f"""You are a highly capable AI assistant designed to solve complex, real-world questions.

REASONING STRATEGY (CRITICAL):
1. **Decompose**: Break complex questions into smaller sub-questions
2. **Plan**: Before using tools, outline your complete strategy
3. **Execute**: Use tools systematically, one step at a time
4. **Verify**: Check each result before proceeding to the next step
5. **Self-correct**: If a tool fails or gives unexpected results, try alternative approaches
6. **Synthesize**: Combine information from multiple sources to form your final answer

FILE HANDLING:
- When questions mention "attached files" or reference files (.mp3, .xlsx, .csv):
  1. First use `list_files` to discover available files
  2. For Excel: use `python_repl` with pandas: `import pandas as pd; df = pd.read_excel('filename.xlsx')`
  3. For CSV: use `python_repl` with pandas: `pd.read_csv('filename.csv')`
  4. For text files: use `read_file` tool

TOOL USAGE BEST PRACTICES:
- Use `calculator` for precise mathematical operations (faster than python_repl)
- Use `wikipedia` for factual knowledge about people, places, events
- Use `tavily_search` for recent information or specific facts
- Use `python_repl` for data analysis, file processing, and complex calculations
- Chain multiple tools when needed (e.g., search → extract info → calculate)

AVAILABLE TOOLS:
{rendered_tools}

RESPONSE FORMAT:
- For tool calls: return JSON with 'name' and 'arguments' keys
- When finished: return JSON with 'name' of 'FINISH'

CRITICAL - ANSWER FORMATTING RULES:
The scoring system is very strict about format. Follow these rules EXACTLY:

1. **For NUMBER answers**:
   - Remove currency symbols ($, €, £)
   - Remove percentage signs (%)
   - Remove commas from large numbers
   - Provide just the number: "1234.56" not "$1,234.56"

2. **For LIST answers** (comma-separated):
   - Use ONLY commas to separate items (or semicolons if specified)
   - NO extra spaces around commas
   - Count must match exactly
   - Order matters!
   - Example: "apple,banana,cherry" NOT "apple, banana, cherry"

3. **For STRING answers**:
   - Be concise - extra words will cause mismatch
   - Capitalization doesn't matter
   - Punctuation doesn't matter
   - Spaces don't matter
   - But be precise with the core answer

4. **For NAMES**:
   - Use full names if asked
   - Use last names only if specified
   - Check the question carefully for format requirements

5. **For CODES** (IOC, airport, etc.):
   - Use exact format requested (uppercase/lowercase)
   - No extra characters

DOUBLE-CHECK YOUR FINAL ANSWER FORMAT BEFORE RETURNING!
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    
    # Create the LLM chains
    _agent_chain = prompt | llm
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based on the conversation history."),
        ("placeholder", "{messages}"),
    ])
    _generation_chain = generation_prompt | llm

def agent_node(state: AgentState) -> dict:
    """Invokes the LLM to decide on the next action."""
    print("\n🤖 [AGENT NODE] Deciding next action...")
    response = _agent_chain.invoke(state)
    
    # Check if there are tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"📝 [AGENT NODE] Requesting {len(response.tool_calls)} tool call(s)")
        for tc in response.tool_calls:
            print(f"   - {tc['name']}")
    else:
        print(f"📝 [AGENT NODE] Response: {response.content[:200]}...")
    
    return {"messages": [response]}

def generation_node(state: AgentState) -> dict:
    """Invokes the LLM to generate a final answer."""
    print("\n✨ [GENERATION NODE] Creating final answer...")
    response = _generation_chain.invoke(state)
    print(f"✅ [GENERATION NODE] Final answer: {response.content[:200]}...")
    return {"messages": [AIMessage(content=response.content)]}

def tool_node(state: AgentState) -> dict:
    """Runs the tools using OpenAI's native tool calling."""
    print("\n🔧 [TOOL NODE] Executing tools...")
    last_message = state["messages"][-1]
    
    # Check if the message has tool_calls (OpenAI's native format)
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("⚠️  [TOOL NODE] No tool calls found")
        return {"messages": []}
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']
        
        print(f"  🛠️  Calling tool: {tool_name}")
        print(f"     Args: {str(tool_args)[:100]}...")
        
        tool_to_call = next((t for t in _tools if t.name == tool_name), None)
        if tool_to_call:
            try:
                observation = tool_to_call.invoke(tool_args)
                result_preview = str(observation)[:150]
                print(f"  ✅ Result: {result_preview}...")
                tool_messages.append(ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call_id
                ))
            except Exception as e:
                print(f"  ❌ Error: {e}")
                tool_messages.append(ToolMessage(
                    content=f"Error: {e}",
                    tool_call_id=tool_call_id
                ))
        else:
            print(f"  ⚠️  Tool '{tool_name}' not found")
    
    print(f"🔧 [TOOL NODE] Executed {len(tool_messages)} tool(s)")
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determines the next node to execute based on OpenAI's tool calls."""
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, go to tools node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("➡️  Routing to: TOOLS")
        return "tools"
    
    # Otherwise, we're done
    print("➡️  Routing to: END")
    return "__end__"

def build_agent_graph() -> StateGraph:
    """Builds the state graph for the agent."""
    # Initialize tools and chains (only happens once)
    _initialize_chains_and_tools()
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

def agent_graph_mermaid() -> str:
    """Returns the LangGraph structure in Mermaid format."""
    graph = build_agent_graph()
    return graph.get_graph().draw_mermaid()

def agent_graph_png_base64(filename: str = "agent_graph.png") -> str | None:
    """Generates a PNG of the agent graph and returns it as a base64 string."""
    graph = build_agent_graph()
    output_path = CACHE_DIR / filename
    try:
        graph.get_graph().draw_png(str(output_path))
    except Exception as exc:
        print(f"Warning: Failed to render agent graph PNG: {exc}")
        return None

    try:
        return base64.b64encode(output_path.read_bytes()).decode("ascii")
    except Exception as exc:
        print(f"Warning: Unable to read rendered graph PNG: {exc}")
        return None
