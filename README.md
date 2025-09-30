---
title: GAIA Benchmark Agent - Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# 🕵🏻‍♂️ GAIA Benchmark AI Agent

A sophisticated LangGraph-based AI agent designed for the GAIA (General AI Assistants) benchmark evaluation. This agent uses GPT-4o with advanced tool-calling capabilities to answer complex multi-step questions involving web search, file analysis, multimedia processing, and reasoning.

## 🎯 Features

### 🧠 **Intelligent Agent**
- LangGraph-based workflow with GPT-4o
- Dual LLM strategy (primary + fallback)
- Up to 50 reasoning iterations per question
- Automatic tool selection and orchestration

### 🛠️ **Comprehensive Tool Suite (15+ tools)**

#### Search & Research
- 🔍 DuckDuckGo Search
- 🔎 Tavily Advanced Search
- 📚 Wikipedia Lookup
- 🎥 YouTube Transcript

#### File Operations
- 📂 List Files (checks current + downloads/)
- 📄 Read Text Files
- 📊 Excel Analysis
- 🐍 Execute Python Scripts
- ⬇️ Download from URLs

#### Multimedia Analysis (Gemini-powered)
- 🎬 **Video Understanding** - Analyzes YouTube videos directly
- 🎵 **Audio Transcription** - Transcribes MP3 files
- 🖼️ **Image Analysis** - Chess positions, diagrams, OCR

#### Computation
- 🧮 Calculator
- 💻 Python REPL

### 💾 **Persistent Caching**
- Answers cached to [agent_cache.json](cci:7://file:///d:/Testers/HuggingFaceFinalAssignment/Final_Assignment_Template/agent_cache.json:0:0-0:0)
- 100% cost savings on repeated questions
- Survives across sessions
- Manually editable

### 🎨 **Answer Cleaning**
- Removes JSON artifacts and code blocks
- Strips explanatory text
- Extracts literal answers only
- Validates non-empty responses

### 🔒 **Robust Error Handling**
- Graceful degradation
- Automatic retries with exponential backoff
- Fallback LLM when primary fails
- Rate limiting protection

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (GPT-4o access)
- Gemini API key (for multimedia)
- Optional: Tavily API key (for advanced search)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Final_Assignment_Template