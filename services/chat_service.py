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
    from services.regret_calculator import get_regret_level, get_regret_explanation
    
    regret_level = get_regret_level(result.regret_score)
    regret_explanation = get_regret_explanation(result.regret_score)
    
    context = f"""Original Decision: {decision_text}

Analysis Results:
- Confidence: {result.confidence:.1%}
- Regret Score: {result.regret_score:.2f} ({regret_level}) - {regret_explanation}
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


def detect_question_type(user_message: str) -> str:
    """
    Detect the type of question being asked.
    
    Args:
        user_message: User's message/question
    
    Returns:
        Question type: "outcome", "emotional", "general"
    """
    user_lower = user_message.lower()
    
    # Check for "what will happen" type questions
    outcome_keywords = ["what will", "what happens", "what if", "outcomes", "results", 
                       "what would", "tell me what", "explain what", "what should i expect"]
    if any(keyword in user_lower for keyword in outcome_keywords):
        return "outcome"
    
    # Check for emotional concerns
    emotional_keywords = ["toxic", "toxicity", "stress", "stressed", "anxious", 
                         "worried", "concerned", "scared", "afraid", "mentally"]
    if any(keyword in user_lower for keyword in emotional_keywords):
        return "emotional"
    
    return "general"


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
    
    # Detect question type
    question_type = detect_question_type(user_message)
    
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
- Understand emotional context and concerns (e.g., if they mention toxicity, stress, fear, etc.)

You have access to the complete analysis including:
- Best case, worst case, and most likely scenarios
- Financial, satisfaction, and risk scores
- Detected risks and their severities
- Overall confidence in the analysis
- Regret Score (proprietary metric for decision regret risk)

Guidelines:
- Reference specific numbers and data from the analysis when relevant
- Help users explore implications of different scenarios
- Acknowledge concerns and provide thoughtful perspectives
- If the user mentions additional context (like "but it is toxic", "I'm stressed", etc.), acknowledge it directly and relate it to the analysis
- Be empathetic and understanding - this is a personal decision
- Use the Regret Score and risk analysis to provide meaningful insights
- If asked about something not in the analysis, say so honestly but try to relate it to what we know
- Keep responses natural and conversational (2-4 sentences is good, but can be longer if needed)
- Don't be robotic - show genuine understanding and care

IMPORTANT: When users share additional context about their situation (like mentioning toxicity, stress, or concerns), acknowledge it directly and help them think through how it relates to the analysis results. Be genuine and empathetic.

"""
    
    # Add conversation history if available
    history_text = ""
    if conversation_history:
        history_text = "\nPrevious conversation:\n"
        for msg in conversation_history[-8:]:  # Last 8 messages for better context
            if msg.get("type") == "analysis_result":
                # Skip the full analysis result display in history, just note it exists
                continue
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            history_text += f"{role}: {content}\n"
        history_text += "\n"
    
    # Detect emotional context
    user_lower = user_message.lower()
    emotional_context = ""
    if any(word in user_lower for word in ["toxic", "toxicity", "terrible", "awful", "bad"]):
        emotional_context = "\nIMPORTANT: The user has mentioned a toxic or negative environment. This is a critical emotional and psychological factor that significantly impacts decision outcomes. Acknowledge this directly and help them understand how toxicity affects long-term satisfaction and well-being beyond just financial metrics.\n"
    elif any(word in user_lower for word in ["stress", "stressed", "anxious", "worried", "concerned"]):
        emotional_context = "\nIMPORTANT: The user is expressing stress or anxiety about the decision. Be empathetic and help them understand how stress factors into the analysis.\n"
    
    # Add specific instructions for "what will happen" questions
    outcome_instructions = ""
    if question_type == "outcome":
        outcome_instructions = """
CRITICAL: The user is asking "what will happen" or similar questions about outcomes.

You MUST:
1. Directly interpret the scenario results from the analysis - don't ask for more information
2. Summarize what each scenario (best case, most likely, worst case) actually shows
3. Use the specific explanations from the analysis provided above
4. Reference concrete numbers (scores, probabilities) from the scenarios
5. Explain what would happen in each scenario based on the data already available
6. Be specific about outcomes - financial, satisfaction, risk scores for each scenario

Do NOT:
- Ask the user to provide more information
- Give generic responses like "it depends" without interpreting the data
- Ignore the scenario explanations already provided

Instead, directly tell them what the analysis shows will happen in each scenario.
"""
    
    # Build the prompt
    prompt = f"""{conversation_context}

{analysis_context}

{history_text}
{emotional_context}
{outcome_instructions}

User's current message: {user_message}

Your response should:
1. Directly acknowledge and address what the user just said - don't give generic responses
2. Reference specific data from the analysis when relevant (Regret Score: {analysis_result.regret_score:.2f}, Confidence: {analysis_result.confidence:.1%}, Risk Score, etc.)
3. Show genuine understanding and empathy
4. Help them think through how their concerns relate to the analysis results
5. Be conversational, natural, and caring - not robotic or scripted

If they mention toxicity, stress, or concerns, address it directly and help them understand the implications. Use the analysis data to support your response, but also show human understanding of their situation.

If they're asking about outcomes ("what will happen"), directly interpret and summarize the scenario results from the analysis above.

Generate your response now:"""

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()
        
        # Clean up response
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]
        
        # Remove any markdown formatting that might interfere
        if reply.startswith('**') or reply.startswith('*'):
            # Keep markdown but ensure it's clean
            pass
        
        return reply
    
    except Exception as e:
        # Better error handling - try to give contextual response even on error
        from services.regret_calculator import get_regret_level, get_regret_explanation
        
        # Include user's message in a way that helps them understand we heard them
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ["toxic", "toxicity", "stress", "stressed", "worried", "concerned", "scared", "afraid"]):
            regret_level = get_regret_level(analysis_result.regret_score)
            return f"I hear you mentioning concerns about {user_message.lower()}. That's a really important factor in your decision - a toxic work environment can significantly impact your well-being and satisfaction, even if the financial aspects look good.\n\nLooking at your analysis, your Regret Score is {analysis_result.regret_score:.2f} ({regret_level}), which suggests {get_regret_explanation(analysis_result.regret_score).lower()} However, the toxicity you're experiencing might not be fully captured in the numbers. Toxic environments can lead to long-term burnout, health issues, and regret even if other factors improve.\n\nWould you like to discuss how this toxicity factor might change your analysis, or explore what your options look like considering this?"
        else:
            # Still try to be helpful with context
            regret_level = get_regret_level(analysis_result.regret_score)
            return f"Based on your analysis, I see your Regret Score is {analysis_result.regret_score:.2f} ({regret_level}), which suggests {get_regret_explanation(analysis_result.regret_score).lower()} \n\nYou mentioned: '{user_message}'. Can you tell me more about how this relates to your decision? I'd like to help you think through this."
