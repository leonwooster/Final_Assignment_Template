"""LangGraph state graph construction for the LLM-powered agent."""

import base64
import json
import os
import time
import random
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from openai import RateLimitError

from .constants import CACHE_DIR
from .tools import tool_classes


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class SimpleRateLimiter:
    """Simple token bucket rate limiter to prevent hitting API limits."""
    
    def __init__(self, calls_per_minute=50):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self):
        """Wait if we're about to exceed rate limit."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # If we're at the limit, wait
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0]) + 1
            if sleep_time > 0:
                print(f"⏳ Rate limiter: waiting {sleep_time:.1f}s to avoid hitting limits...")
                time.sleep(sleep_time)
                self.call_times = []
        
        # Record this call
        self.call_times.append(time.time())


# Global variables that will be initialized in build_agent_graph()
_tools = None
_agent_chain = None
_generation_chain = None
_primary_llm = None
_fallback_llm = None
_rate_limiter = SimpleRateLimiter(calls_per_minute=40)  # Conservative limit

def _call_llm_with_retry(chain, state, max_retries=5):
    """
    Call LLM with exponential backoff retry logic.
    Falls back to cheaper model if primary keeps failing.
    """
    for attempt in range(max_retries):
        try:
            # Wait if we're approaching rate limits
            _rate_limiter.wait_if_needed()
            return chain.invoke(state)
        except RateLimitError as e:
            # Extract wait time from error if available
            wait_time = min(60, (2 ** attempt) + random.random())
            print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
            print(f"   Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
        except Exception as e:
            # For other errors, don't retry
            print(f"❌ LLM error: {e}")
            raise
    
    # If all retries failed, try fallback model
    print("🔄 All retries exhausted, switching to fallback model (gpt-4o-mini)...")
    try:
        # Rebuild chain with fallback LLM
        if _fallback_llm is not None:
            fallback_chain = chain.first | _fallback_llm
            return fallback_chain.invoke(state)
    except Exception as e:
        print(f"❌ Fallback model also failed: {e}")
        raise
    
    raise RuntimeError("All retry attempts and fallback failed")

def _initialize_chains_and_tools():
    """Initialize the tools and LLM chains. Called once when building the graph."""
    global _tools, _agent_chain, _generation_chain, _primary_llm, _fallback_llm
    
    if _tools is not None:
        return  # Already initialized
    
    # Initialize PRIMARY LLM (gpt-4o)
    print("🔧 Initializing primary LLM: gpt-4o")
    _primary_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        verbose=True,
        request_timeout=60  # 60 second timeout
    )
    
    # Initialize FALLBACK LLM (gpt-4o-mini - cheaper, faster)
    print("🔧 Initializing fallback LLM: gpt-4o-mini")
    _fallback_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        verbose=True,
        request_timeout=60
    )
    
    llm = _primary_llm
    
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

FILE HANDLING - CRITICAL:
⚠️ **Files mentioned as "attached" are ALREADY in the current directory!**
- When question says "attached Excel file", "attached image", "attached .mp3" - use `list_files` to find them
- Files are pre-downloaded before you start, so they WILL be in current directory
- **NEVER** ask for URLs for "attached" files - they're already there!
- Workflow:
  1. Use `list_files` to see what's available
  2. Find the relevant file (Excel, image, mp3, etc.)
  3. Process it with appropriate tool:
     - Excel (.xlsx, .xls): use `read_excel` tool OR `python_repl` with pandas
     - CSV: use `python_repl` with pandas: `pd.read_csv('filename.csv')`
     - Text/Python files: use `read_file` tool
     - Images: use python_repl with PIL/OpenCV if analysis needed
     - MP3: check for transcript file or mention you cannot process audio directly

MULTIMEDIA HANDLING:
- For YouTube videos: use `youtube_transcript` tool with the video URL
- For web URLs: use `download_file` if you need to download something from the internet

TOOL USAGE BEST PRACTICES:
- Use `calculator` for precise mathematical operations (faster than python_repl)
- Use `wikipedia` for factual knowledge about people, places, events
- Use `tavily_search` for recent information or specific facts
- Use `youtube_transcript` for YouTube video content analysis
- Use `read_excel` for quick Excel file inspection
- Use `python_repl` for complex data analysis and calculations
- Chain multiple tools when needed (e.g., search → extract info → calculate)

AVAILABLE TOOLS:
{rendered_tools}

RESPONSE FORMAT:
- For tool calls: return JSON with 'name' and 'arguments' keys
- When finished: return JSON with 'name' of 'FINISH'

⚠️ CRITICAL - PROVIDE ONLY THE FINAL ANSWER ⚠️
DO NOT include explanations, reasoning, or extra text in your final answer.
Examples:
- Question: "How many albums?" → Answer: "2" (NOT "Mercedes Sosa published 2 albums...")
- Question: "What city?" → Answer: "Paris" (NOT "The city is Paris")
- Question: "Total sales?" → Answer: "1234.56" (NOT "The total sales are $1,234.56")

BE EXTREMELY CONCISE. The scoring system only wants the literal answer.

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
    """Invokes the LLM to decide on the next action with retry logic."""
    print("\n🤖 [AGENT NODE] Deciding next action...")
    
    # Use retry logic
    response = _call_llm_with_retry(_agent_chain, state)
    
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
