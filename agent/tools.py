"""LangGraph tools for the agent."""

import os
from pathlib import Path
from typing import Optional
import time

from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonAstREPLTool
from langchain_tavily import TavilySearch
from youtube_transcript_api import YouTubeTranscriptApi
import re as regex
import requests
from urllib.parse import urlparse


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


class ExcelReaderTool(BaseTool):
    """Tool for reading and analyzing Excel files."""
    
    name: str = "read_excel"
    description: str = "Reads an Excel file and returns its contents as a pandas DataFrame. Input should be the file path to the Excel file (.xlsx or .xls)."
    
    def _run(self, file_path: str) -> str:
        """Read Excel file and return summary."""
        try:
            import pandas as pd
            
            path = Path(file_path)
            if not path.exists():
                return f"File '{file_path}' does not exist."
            
            # Read the Excel file
            df = pd.read_excel(path)
            
            # Return a summary
            result = f"Excel file loaded successfully.\n"
            result += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            result += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            result += f"First few rows:\n{df.head().to_string()}\n\n"
            result += f"Data types:\n{df.dtypes.to_string()}\n\n"
            result += f"Summary statistics:\n{df.describe().to_string()}"
            
            return result
        except Exception as e:
            return f"Error reading Excel file: {str(e)}"
    
    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)


class DownloadFileTool(BaseTool):
    """Tool for downloading files from URLs."""
    
    name: str = "download_file"
    description: str = "Downloads a file from a URL and saves it to the current directory. Input should be the URL of the file to download. Returns the local file path."
    
    def _run(self, url: str) -> str:
        """Download file from URL."""
        try:
            # Parse URL to get filename
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            
            # If no filename in URL, generate one
            if not filename or '.' not in filename:
                # Try to get from Content-Disposition header
                response = requests.head(url, allow_redirects=True, timeout=10)
                if 'Content-Disposition' in response.headers:
                    content_disp = response.headers['Content-Disposition']
                    if 'filename=' in content_disp:
                        filename = content_disp.split('filename=')[1].strip('"\'')
                
                # Still no filename? Generate one based on content type
                if not filename or '.' not in filename:
                    content_type = response.headers.get('Content-Type', '')
                    ext = '.bin'
                    if 'image' in content_type:
                        ext = '.png' if 'png' in content_type else '.jpg'
                    elif 'excel' in content_type or 'spreadsheet' in content_type:
                        ext = '.xlsx'
                    elif 'pdf' in content_type:
                        ext = '.pdf'
                    filename = f"downloaded_file{ext}"
            
            # Download the file
            print(f"📥 Downloading: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to current directory
            filepath = Path(filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"✅ Downloaded: {filename} ({file_size} bytes)")
            
            return f"File downloaded successfully: {filename} ({file_size} bytes)"
        except Exception as e:
            return f"Error downloading file: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        """Async version."""
        return self._run(url)


class YouTubeTranscriptTool(BaseTool):
    """Tool for getting transcripts from YouTube videos."""
    
    name: str = "youtube_transcript"
    description: str = "Gets the transcript/captions from a YouTube video. Input should be either a YouTube URL or video ID."
    
    def _run(self, video_input: str) -> str:
        """Get YouTube transcript."""
        try:
            # Extract video ID from URL if needed
            video_id = video_input
            if 'youtube.com' in video_input or 'youtu.be' in video_input:
                # Extract video ID from various YouTube URL formats
                patterns = [
                    r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
                    r'(?:embed\/)([0-9A-Za-z_-]{11})',
                    r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
                ]
                for pattern in patterns:
                    match = regex.search(pattern, video_input)
                    if match:
                        video_id = match.group(1)
                        break
            
            # Get transcript using the correct API
            try:
                # Try to get transcript (auto-generated or manual)
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                # Try any available language
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = next(iter(transcript_list))
                    transcript_data = transcript.fetch()
                except:
                    return f"Error: No transcript available for video {video_id}"
            
            # Format transcript
            full_transcript = "\n".join([f"[{item['start']:.1f}s] {item['text']}" for item in transcript_data])
            
            return f"YouTube Transcript for video {video_id}:\n\n{full_transcript}"
        except Exception as e:
            return f"Error getting YouTube transcript: {str(e)}"
    
    async def _arun(self, video_input: str) -> str:
        """Async version."""
        return self._run(video_input)


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
    ExcelReaderTool,
    DownloadFileTool,
    YouTubeTranscriptTool,
    CalculatorTool,
    create_wikipedia_tool  # This returns an instance, not a class
]   
