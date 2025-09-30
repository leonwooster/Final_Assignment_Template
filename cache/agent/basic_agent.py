"""A LangGraph-based agent implementation."""

import re
import sys
import json
from pathlib import Path
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage

from .graph import AgentState, build_agent_graph


def ensure_valid_answer(answer: str) -> str:
    """Ensure answer is never None or empty."""
    if not answer or not isinstance(answer, str) or answer.strip() == "":
        return "Unable to determine answer"
    return answer.strip()


class TeeOutput:
    """Redirect stdout/stderr to both console and file."""
    def __init__(self, file_path, mode='a'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.terminal = sys.stdout if mode == 'a' else sys.stderr
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


class BasicAgent:
    """A LangGraph-powered agent that uses tools to answer questions."""

    def __init__(self, log_to_file=True, use_cache=True, cache_file="agent_cache.json") -> None:
        """Initialize the agent with the compiled graph."""
        self.graph = build_agent_graph()
        self.log_file = None
        self.use_cache = use_cache
        self.cache_file = Path(cache_file)
        self.answer_cache = {}  # Cache for question -> answer mapping
        
        # Load cache from disk if it exists
        if self.use_cache:
            self._load_cache()
        
        # Set up logging to file
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"agent_run_{timestamp}.log"
            self.log_file = TeeOutput(log_filename, 'w')
            sys.stdout = self.log_file
            print(f"📝 Logging to: {log_filename}\n")
            if self.use_cache and self.answer_cache:
                print(f"💾 Loaded {len(self.answer_cache)} cached answers from {self.cache_file}\n")

    def _load_cache(self):
        """Load answer cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.answer_cache = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load cache from {self.cache_file}: {e}")
            self.answer_cache = {}
    
    def _save_cache(self):
        """Save answer cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.answer_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not save cache to {self.cache_file}: {e}")

    def _clean_answer(self, answer: str, question: str) -> str:
        """
        Clean the answer based on GAIA scoring rules.
        Aggressively removes explanatory text to provide only the literal answer.
        """
        answer = answer.strip()
        
        # Remove JSON formatting and code blocks
        if answer.startswith('```'):
            # Extract content from code blocks
            lines = answer.split('\n')
            answer = '\n'.join([l for l in lines if not l.startswith('```')])
            answer = answer.strip()
        
        # Remove JSON structures like {"name":"FINISH","answer":"value"}
        if answer.startswith('{') and ('"name"' in answer or '"FINISH"' in answer):
            try:
                import json
                # Try to parse as JSON
                parsed = json.loads(answer)
                # Extract the actual answer value from various possible keys
                for key in ['answer', 'arguments', 'vegetables', 'surname', 'value', 'result', 'submitted_answer']:
                    if key in parsed and parsed[key] and parsed[key] != "FINISH":
                        answer = str(parsed[key])
                        break
                # If still has "name" field, it's probably still JSON - extract any non-name value
                if isinstance(parsed, dict) and 'name' in parsed:
                    for key, value in parsed.items():
                        if key != 'name' and key != 'FINISH' and value and value != "FINISH":
                            answer = str(value)
                            break
            except:
                pass
        
        # Remove common prefixes and explanatory phrases
        patterns_to_remove = [
            r'^(the answer is|answer:|final answer:|thus,|therefore,|so,|hence,)\s*',
            r'^(the\s+)?(correct\s+)?(number|city|country|name|value|total|result)\s+(is|are|was|were)\s*',
            r'^\d+\.\s*',  # Remove leading numbers like "1. " or "2. "
            r'^[-•]\s*',   # Remove bullet points
        ]
        
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            answer = answer.strip()
        
        # If answer contains multiple sentences, try to extract just the key info
        sentences = answer.split('.')
        if len(sentences) > 1:
            # Look for the shortest sentence that contains key info
            for sent in sentences:
                sent = sent.strip()
                # If it's short and contains a number or key word, use it
                if len(sent) < 50 and (any(char.isdigit() for char in sent) or len(sent.split()) <= 5):
                    answer = sent
                    break
        
        # Remove trailing explanations in parentheses
        answer = re.sub(r'\s*\([^)]*\)\s*$', '', answer)
        
        # If the question asks for a comma-separated list, ensure no spaces after commas
        if 'comma' in question.lower() and ('list' in question.lower() or 'separated' in question.lower()):
            answer = re.sub(r',\s+', ',', answer)
        
        # Clean numbers: remove currency symbols and commas
        if len(answer.split()) <= 5:  # Short answer, likely a number
            if any(char.isdigit() for char in answer):
                cleaned = answer
                for symbol in ['$', '€', '£', '¥', '%', ',']:
                    cleaned = cleaned.replace(symbol, '')
                
                # If after cleaning it's still a valid number, use the cleaned version
                try:
                    float(cleaned.strip())
                    answer = cleaned.strip()
                except ValueError:
                    pass  # Not a pure number, keep original
        
        # Final cleanup: remove quotes if they wrap the entire answer
        answer = answer.strip('"\'')
        
        return answer

    def __call__(self, question: str) -> str:
        """Invoke the agent with a question and return the answer."""
        try:
            print("\n" + "="*80)
            print(f"📋 QUESTION: {question[:150]}...")
            print("="*80)
            
            # Check cache first
            if self.use_cache and question in self.answer_cache:
                cached_answer = self.answer_cache[question]
                print("\n💾 Using cached answer (no LLM call!)")
                print(f"\n🎯 FINAL ANSWER: {cached_answer}")
                print("="*80 + "\n")
                return cached_answer
            
            # Create the initial state with the user's question
            state: AgentState = {"messages": [HumanMessage(content=question)]}
            
            # Run the graph with increased recursion limit
            print("\n🚀 Starting agent execution...")
            result = self.graph.invoke(state, config={"recursion_limit": 50})
            
            # Extract the final answer from the last AI message
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    raw_answer = message.content
                    # Clean the answer based on GAIA scoring rules
                    cleaned_answer = self._clean_answer(raw_answer, question)
                    # Ensure answer is never empty
                    validated_answer = ensure_valid_answer(cleaned_answer)
                    
                    # Cache the answer and save to disk
                    if self.use_cache:
                        self.answer_cache[question] = validated_answer
                        self._save_cache()  # Persist to disk immediately
                    
                    print(f"\n🎯 FINAL ANSWER: {validated_answer}")
                    print("="*80 + "\n")
                    return validated_answer
            
            print("\n⚠️  No answer found")
            print("="*80 + "\n")
            return ensure_valid_answer("")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print("="*80 + "\n")
            return ensure_valid_answer(f"Agent failed with error: {e}")
