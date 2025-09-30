"""LangGraph tools for the agent."""

import os
from pathlib import Path
from typing import Optional
import time

from langchain.tools import BaseTool, WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonAstREPLTool
from langchain_tavily import TavilySearch


class ListFilesTool(BaseTool):
    """Tool to list files in the current directory or a specified directory."""
    
    name: str = "list_files"
    description: str = "Lists all files in the current directory or a specified directory. Input should be a directory path (optional, defaults to current directory)."
    
    def _run(self, directory: str = ".") -> str:
        """List files in the specified directory."""
        try:
            path = Path(directory)
            if not path.exists():
                return f"Directory '{directory}' does not exist."
            
            files = []
            for item in path.iterdir():
                if item.is_file():
                    files.append(f"{item.name} ({item.stat().st_size} bytes)")
            
            if not files:
                return f"No files found in '{directory}'."
            
            return "Files found:\n" + "\n".join(files)
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    async def _arun(self, directory: str = ".") -> str:
        """Async version."""
        return self._run(directory)


class ReadFileTool(BaseTool):
    """Tool to read the contents of a text file."""
    
    name: str = "read_file"
    description: str = "Reads the contents of a text file. Input should be the file path."
    
    def _run(self, file_path: str) -> str:
        """Read the file contents."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"File '{file_path}' does not exist."
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations safely."""
    
    name: str = "calculator"
    description: str = "Useful for mathematical calculations. Input should be a mathematical expression as a string (e.g., '2 + 2', '(5 * 3) / 2')."
    
    def _run(self, expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # Remove any potentially dangerous operations
            if any(dangerous in expression.lower() for dangerous in ['import', 'exec', 'eval', '__']):
                return "Error: Expression contains forbidden operations."
            
            # Evaluate using Python's eval with restricted namespace
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async version."""
        return self._run(expression)


# Create Wikipedia tool wrapper
def create_wikipedia_tool():
    """Create a Wikipedia search tool."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
    return WikipediaQueryRun(api_wrapper=api_wrapper)


tool_classes = [
    DuckDuckGoSearchRun, 
    TavilySearch, 
    PythonAstREPLTool,
    ListFilesTool,
    ReadFileTool,
    CalculatorTool,
    create_wikipedia_tool  # This returns an instance, not a class
]
