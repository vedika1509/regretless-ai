"""Service for conversational interaction with analysis results."""
import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from models.scenario import AnalysisResult

load_dotenv()


def initialize_gemini() -> genai.GenerativeModel:
    """Initialize Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')


def format_analysis_context(result: AnalysisResult, decision_text: str) -> str:
    """
    Format analysis results as context for conversation.
    
    Args:
        result: AnalysisResult object
        decision_text: Original decision text
    
    Returns:
        Formatted context string
    """
    context = f"""Original Decision: {decision_text}

Analysis Results:
- Confidence: {result.confidence:.1%}
- Simulations Run: {result.simulation_count:,}

Scenarios:
"""
    
    scenario_names = {
        "best_case": "Best Case",
        "worst_case": "Worst Case",
        "most_likely": "Most Likely"
    }
    
    for scenario_type, scenario in result.scenarios.items():
        scenario_name = scenario_names.get(scenario_type, scenario_type)
        context += f"""
{scenario_name} (Probability: {scenario.probability:.1%}):
- Financial Score: {scenario.outcomes.financial_score:.2f}
- Satisfaction Score: {scenario.outcomes.satisfaction_score:.2f}
- Risk Score: {scenario.outcomes.risk_score:.2f}
- Overall Score: {scenario.outcomes.overall_score:.2f}
- Explanation: {scenario.explanation}
"""
    
    if result.risks:
        context += "\nDetected Risks:\n"
        for risk in result.risks:
            context += f"- {risk.risk_type.replace('_', ' ').title()} ({risk.severity}): {risk.description}\n"
    
    return context


def generate_chat_response(
    user_message: str,
    analysis_result: AnalysisResult,
    decision_text: str,
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    Generate conversational response based on user message and analysis context.
    
    Args:
        user_message: User's message/question
        analysis_result: AnalysisResult object
        decision_text: Original decision text
        conversation_history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]
    
    Returns:
        Assistant's response
    """
    model = initialize_gemini()
    
    # Format analysis context
    analysis_context = format_analysis_context(analysis_result, decision_text)
    
    # Build conversation context
    conversation_context = """You are Regretless AI, a helpful decision analysis assistant. You've just provided an analysis of a user's decision using probabilistic simulation.

Your role:
- Help users understand their decision analysis results
- Answer questions about scenarios, risks, and confidence scores
- Provide empathetic, clear, and practical guidance
- Help users reason through their decision using the analysis data
- Be conversational, friendly, and supportive

You have access to the complete analysis including:
- Best case, worst case, and most likely scenarios
- Financial, satisfaction, and risk scores
- Detected risks and their severities
- Overall confidence in the analysis

Guidelines:
- Reference specific numbers and data from the analysis when relevant
- Help users explore implications of different scenarios
- Acknowledge concerns and provide thoughtful perspectives
- If asked about something not in the analysis, say so honestly
- Keep responses concise (2-4 sentences) unless asked to elaborate
- Use natural, conversational language

"""
    
    # Add conversation history if available
    history_text = ""
    if conversation_history:
        history_text = "\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"
    
    # Build the prompt
    prompt = f"""{conversation_context}

{analysis_context}

{history_text}

User's current message: {user_message}

Provide a helpful, conversational response that addresses the user's question or comment. Reference specific data from the analysis when relevant."""

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()
        
        # Clean up response
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        
        return reply
    
    except Exception as e:
        # Fallback response
        return f"I understand your question about the analysis. Let me help you think through this decision. Based on the simulation results, we can explore different aspects of your decision. Could you tell me what specific aspect you'd like to discuss?"
