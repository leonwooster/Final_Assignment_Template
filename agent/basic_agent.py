"""A LangGraph-based agent implementation."""

import re
from langchain_core.messages import AIMessage, HumanMessage

from .graph import AgentState, build_agent_graph


class BasicAgent:
    """A LangGraph-powered agent that uses tools to answer questions."""

    def __init__(self) -> None:
        """Initialize the agent with the compiled graph."""
        self.graph = build_agent_graph()

    def _clean_answer(self, answer: str, question: str) -> str:
        """
        Clean the answer based on GAIA scoring rules.
        This helps format answers correctly for the evaluation system.
        """
        answer = answer.strip()
        
        # Detect if this is likely a number answer
        # Remove common phrases that might wrap numbers
        answer = re.sub(r'^(the answer is|answer:|final answer:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # If the question asks for a comma-separated list, ensure no spaces after commas
        if 'comma' in question.lower() and 'list' in question.lower():
            # Remove spaces after commas
            answer = re.sub(r',\s+', ',', answer)
        
        # If it looks like a number with currency or percentage, clean it
        # But only if it's a simple number (not part of a longer text)
        if len(answer.split()) <= 3:  # Short answer, likely a number
            # Check if it contains number-like characters
            if any(char.isdigit() for char in answer):
                # Remove currency symbols and commas from what looks like a number
                cleaned = answer
                for symbol in ['$', '€', '£', '¥', '%', ',']:
                    cleaned = cleaned.replace(symbol, '')
                
                # If after cleaning it's still a valid number, use the cleaned version
                try:
                    float(cleaned.strip())
                    answer = cleaned.strip()
                except ValueError:
                    pass  # Not a pure number, keep original
        
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
