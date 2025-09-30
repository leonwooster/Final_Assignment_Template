# Agent Architecture Documentation

## Overview

This is a LangGraph-based AI agent designed for the GAIA (General AI Assistants) benchmark evaluation. The agent uses GPT-4o/GPT-4o-mini with tool-calling capabilities to answer complex multi-step questions involving web search, file analysis, multimedia processing, and reasoning.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                             │
│                    (20 GAIA Questions)                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                          app.py                                  │
│  • Fetches questions from API                                    │
│  • Downloads attached files (Excel, MP3, images, Python)         │
│  • Saves files to downloads/ directory                           │
│  • Calls BasicAgent for each question                            │
│  • Submits answers to evaluation API                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      BasicAgent                                  │
│                   (agent/basic_agent.py)                         │
│                                                                   │
│  1. Check Cache (agent_cache.json)                               │
│     └─ If cached: Return answer instantly ✅                     │
│                                                                   │
│  2. If not cached:                                               │
│     └─ Invoke LangGraph workflow                                 │
│                                                                   │
│  3. Clean & validate answer                                      │
│     └─ Remove JSON, code blocks, explanations                    │
│                                                                   │
│  4. Cache answer to disk                                         │
│     └─ Save to agent_cache.json for future use                   │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                            │
│                     (agent/graph.py)                             │
│                                                                   │
│  ┌──────────────┐                                                │
│  │ Agent Node   │ ← Decides next action                          │
│  │ (GPT-4o)     │   • Analyze question                           │
│  └──────┬───────┘   • Choose tool(s)                             │
│         │           • Generate response                           │
│         ↓                                                         │
│  ┌──────────────┐                                                │
│  │ Tools Node   │ ← Executes tools                               │
│  │              │   • Search, calculate, read files              │
│  └──────┬───────┘   • Returns results                            │
│         │                                                         │
│         ↓                                                         │
│  ┌──────────────┐                                                │
│  │ Agent Node   │ ← Processes results                            │
│  │ (GPT-4o)     │   • Analyzes tool output                       │
│  └──────┬───────┘   • Decides: more tools or final answer?       │
│         │                                                         │
│         └─────────→ Loop (max 50 iterations)                     │
│                                                                   │
│  Final Answer → Return to BasicAgent                             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. **app.py** - Main Application
**Responsibilities:**
- Fetch questions from evaluation API
- Download attached files from `/files/{task_id}` endpoint
- Orchestrate agent execution for all questions
- Submit answers to evaluation API
- Display results

**Key Functions:**
- `run_and_submit_all()` - Main evaluation loop
- File download with error handling
- Results aggregation and submission

### 2. **agent/basic_agent.py** - Agent Wrapper
**Responsibilities:**
- Manage agent lifecycle
- Implement caching system (persistent to disk)
- Clean and validate answers
- Logging to file

**Key Features:**
- **Persistent Caching:** Saves answers to `agent_cache.json`
- **Answer Cleaning:** Removes JSON, code blocks, explanations
- **Validation:** Ensures no empty answers submitted
- **Logging:** All output saved to timestamped log files

**Cache System:**
```python
{
  "question_text": "answer",
  "How many albums...": "4",
  "What is 2+2?": "4"
}
```

### 3. **agent/graph.py** - LangGraph Workflow
**Responsibilities:**
- Define agent workflow (nodes and edges)
- Initialize LLM chains (primary + fallback)
- Initialize and manage tools
- Route between agent and tools nodes

**Workflow Structure:**
```
START → Agent Node → [Tools Node] → Agent Node → END
         ↑_______________|
         (loop until answer found or max iterations)
```

**Key Components:**
- `agent_node()` - LLM decision making
- `tool_node()` - Tool execution
- `should_continue()` - Routing logic
- System prompt with detailed instructions

**LLM Configuration:**
- **Primary:** GPT-4o (with tools)
- **Fallback:** GPT-4o-mini (with tools)
- **Recursion Limit:** 50 iterations
- **Rate Limiting:** Exponential backoff (5 retries)

### 4. **agent/tools.py** - Tool Implementations
**Responsibilities:**
- Implement all tools available to the agent
- Handle file path resolution (current dir + downloads/)
- Integrate with external APIs (Gemini, search engines)

**Available Tools:**

#### Search & Research (5 tools)
- `duckduckgo_search` - Web search
- `tavily_search` - Advanced web search
- `wikipedia` - Wikipedia lookup
- `youtube_transcript` - Get YouTube transcripts
- `arxiv_search` - Academic paper search

#### File Operations (5 tools)
- `list_files` - List files in current/downloads directory
- `read_file` - Read text files
- `read_excel` - Read and analyze Excel files
- `download_file` - Download files from URLs
- `execute_python_file` - Run Python scripts

#### Multimedia Analysis (3 tools - Gemini-powered)
- `understand_video` - Analyze YouTube videos
- `understand_audio` - Transcribe and analyze MP3/audio
- `analyze_image` - Analyze images (chess, diagrams, text)

#### Computation (2 tools)
- `calculator` - Safe math evaluation
- `python_repl` - Execute Python code

**File Path Resolution:**
All file tools use `find_file()` helper that checks:
1. Current directory
2. `downloads/` directory
3. Returns best match or downloads path

## Data Flow

### Question Processing Flow

```
1. API Request
   └─ GET /questions
   └─ Returns: [{task_id, question, Level, file_name}, ...]

2. File Download (if file_name exists)
   └─ GET /files/{task_id}
   └─ Save to: downloads/{file_name}

3. Agent Invocation
   ├─ Check cache
   │  └─ If hit: Return cached answer (0 LLM calls)
   │
   └─ If miss:
      ├─ Create initial state with question
      ├─ Invoke LangGraph workflow
      │  ├─ Agent decides action
      │  ├─ Execute tools
      │  ├─ Agent processes results
      │  └─ Loop until answer or max iterations
      │
      ├─ Extract answer from final message
      ├─ Clean answer (remove JSON, explanations)
      ├─ Validate answer (ensure not empty)
      └─ Cache to disk

4. Answer Submission
   └─ POST /submit
   └─ Body: {username, answers: [{task_id, submitted_answer}]}
```

## Tool Execution Flow

```
Agent Node (GPT-4o)
  ↓
Decides: "I need to use list_files tool"
  ↓
Tool Node
  ├─ Finds tool by name
  ├─ Validates parameters
  ├─ Executes tool._run()
  │  └─ Example: list_files()
  │     ├─ Check current directory
  │     ├─ Check downloads/ directory
  │     └─ Return: "Files found:\n./app.py\ndownloads/data.xlsx"
  └─ Returns ToolMessage with result
  ↓
Agent Node (GPT-4o)
  ├─ Receives tool output
  ├─ Analyzes results
  └─ Decides: Use another tool OR provide final answer
```

## Key Design Decisions

### 1. **Persistent Caching**
**Why:** Reduce costs and enable fast re-runs
**How:** JSON file on disk, loaded at startup, saved after each answer
**Benefit:** 100% cost savings on repeated questions

### 2. **File Path Resolution**
**Why:** Files can be in current directory or downloads/
**How:** `find_file()` helper checks both locations
**Benefit:** Agent doesn't need to know exact file location

### 3. **Gemini for Multimedia**
**Why:** GPT-4o doesn't support direct video/audio analysis
**How:** Upload files to Gemini API, get analysis
**Benefit:** Can handle YouTube videos, MP3 files, images

### 4. **Answer Cleaning Pipeline**
**Why:** LLMs often return verbose explanations or JSON
**How:** Multi-stage cleaning (JSON removal, pattern matching, validation)
**Benefit:** Clean, concise answers that match expected format

### 5. **Dual LLM Strategy**
**Why:** Reliability and cost optimization
**How:** Primary (GPT-4o) with fallback (GPT-4o-mini)
**Benefit:** Continues working if primary fails

### 6. **Tool-First Architecture**
**Why:** Many questions require external data
**How:** Rich tool suite with 15+ specialized tools
**Benefit:** Can handle diverse question types

## Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-...           # Required for GPT-4o
GEMINI_API_KEY=...              # Required for video/audio/image analysis
TAVILY_API_KEY=...              # Optional for advanced search
HF_TOKEN=...                    # For HuggingFace API access
```

### Configurable Parameters

**In BasicAgent:**
- `log_to_file` - Enable/disable logging (default: True)
- `use_cache` - Enable/disable caching (default: True)
- `cache_file` - Cache file path (default: "agent_cache.json")

**In LangGraph:**
- `recursion_limit` - Max iterations (default: 50)
- `temperature` - LLM temperature (default: 0.0)
- `max_retries` - Rate limit retries (default: 5)

## File Structure

```
Final_Assignment_Template/
├── app.py                      # Main application
├── agent/
│   ├── __init__.py
│   ├── basic_agent.py          # Agent wrapper with caching
│   ├── graph.py                # LangGraph workflow
│   └── tools.py                # Tool implementations
├── downloads/                  # Downloaded files (gitignored)
│   ├── file1.xlsx
│   ├── audio.mp3
│   └── image.png
├── agent_cache.json            # Persistent cache (gitignored)
├── agent_run_*.log             # Log files (gitignored)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .gitignore
├── ARCHITECTURE.md             # This file
└── README.md                   # User documentation
```

## Performance Characteristics

### Typical Question Processing Time
- **Simple (cached):** < 0.1 seconds
- **Simple (web search):** 2-5 seconds
- **Medium (file analysis):** 5-15 seconds
- **Complex (multi-step):** 15-60 seconds
- **Multimedia (video/audio):** 30-120 seconds

### LLM Token Usage (per question)
- **Simple:** 500-2,000 tokens
- **Medium:** 2,000-8,000 tokens
- **Complex:** 8,000-20,000 tokens

### Cost Estimates (GPT-4o)
- **Per question (avg):** $0.01-0.05
- **20 questions:** $0.20-1.00
- **With caching (re-runs):** $0.00

## Error Handling

### Graceful Degradation
1. **Cache file corrupted:** Start with empty cache
2. **File download fails:** Continue without file, agent handles gracefully
3. **Tool execution fails:** Return error message, agent tries alternative
4. **LLM rate limit:** Exponential backoff, retry up to 5 times
5. **Primary LLM fails:** Fallback to GPT-4o-mini
6. **Recursion limit hit:** Return best answer so far

### Validation
- All answers validated (never empty)
- File paths validated before access
- API responses validated before processing
- Tool parameters validated before execution

## Testing & Development

### Local Testing
```bash
# Run full evaluation
python app.py

# Check logs
tail -f agent_run_*.log

# View cache
cat agent_cache.json

# Clear cache for fresh run
rm agent_cache.json
```

### Debugging
- All tool calls logged with arguments and results
- Agent reasoning logged at each step
- Errors logged with full stack traces
- Cache hits/misses logged

## Future Enhancements

### Potential Improvements
1. **Pattern-based answering** - Skip LLM for simple questions
2. **Parallel tool execution** - Run independent tools simultaneously
3. **Smarter caching** - Fuzzy matching for similar questions
4. **Cost tracking** - Log token usage and costs
5. **A/B testing** - Compare different prompts/strategies
6. **Streaming responses** - Show progress in real-time

### Scalability Considerations
- Cache can grow large (consider size limits or TTL)
- Multiple concurrent runs need separate cache files
- Rate limiting may need adjustment for production
- Consider database instead of JSON for large-scale caching

## Dependencies

### Core
- `langchain` - LLM framework
- `langgraph` - Workflow orchestration
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools

### Tools
- `google-generativeai` - Gemini API
- `tavily-python` - Advanced search
- `duckduckgo-search` - Web search
- `youtube-transcript-api` - YouTube transcripts
- `pandas` - Data analysis
- `openpyxl` - Excel files

### Utilities
- `requests` - HTTP requests
- `python-dotenv` - Environment variables
- `gradio` - Web UI (optional)

## Security Considerations

### API Keys
- Stored in `.env` file (gitignored)
- Never hardcoded in source
- Loaded via `python-dotenv`

### Code Execution
- `python_repl` uses AST-based REPL (safer than eval)
- `execute_python_file` runs in subprocess with timeout
- No shell injection vulnerabilities

### File Access
- All file operations use Path validation
- No arbitrary file system access
- Downloads isolated to `downloads/` directory

## Monitoring & Observability

### Logs
- Timestamped log files for each run
- Structured logging with emojis for easy parsing
- Tool calls logged with full context
- Errors logged with stack traces

### Metrics (available in logs)
- Questions processed
- Cache hit rate
- Tool usage frequency
- LLM calls per question
- Execution time per question
- Error rate

---

**Version:** 1.0  
**Last Updated:** 2025-09-30  
**Author:** Leon Woo  
**License:** MIT
