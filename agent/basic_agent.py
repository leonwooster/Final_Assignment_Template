"""A LangGraph-based agent implementation."""

import re
import sys
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage

from .graph import AgentState, build_agent_graph


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

    def __init__(self, log_to_file=True) -> None:
        """Initialize the agent with the compiled graph."""
        self.graph = build_agent_graph()
        self.log_file = None
        
        # Set up logging to file
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"agent_run_{timestamp}.log"
            self.log_file = TeeOutput(log_filename, 'w')
            sys.stdout = self.log_file
            print(f"📝 Logging to: {log_filename}\n")

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
        if answer.startswith('{') and '"name"' in answer and '"FINISH"' in answer:
            try:
                import json
                parsed = json.loads(answer)
                # Extract the actual answer value from various possible keys
                for key in ['answer', 'vegetables', 'surname', 'value', 'result']:
                    if key in parsed:
                        answer = parsed[key]
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
            
            # Create the initial state with the user's question
            state: AgentState = {"messages": [HumanMessage(content=question)]}
            
            # Run the graph
            print("\n🚀 Starting agent execution...")
            result = self.graph.invoke(state)
            
            # Extract the final answer from the last AI message
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    raw_answer = message.content
                    # Clean the answer based on GAIA scoring rules
                    cleaned_answer = self._clean_answer(raw_answer, question)
                    print(f"\n🎯 FINAL ANSWER: {cleaned_answer}")
                    print("="*80 + "\n")
                    return cleaned_answer
            
            print("\n⚠️  No answer found")
            print("="*80 + "\n")
            return "No answer found."
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print("="*80 + "\n")
            return f"Agent failed with error: {e}"
