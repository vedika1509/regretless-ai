"""Service for conversational interaction with analysis results."""
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from models.scenario import AnalysisResult
from services.llm.groq_client import safe_chat
from services.llm.guardrails import extract_number_tokens, redact_invented_numbers
from services.safety import assess_safety

load_dotenv()


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
    conversation_history: List[Dict[str, str]] = None,
    interpretation: Optional[Dict[str, Any]] = None,
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
    # Detect question type
    question_type = detect_question_type(user_message)
    
    # Format analysis context
    analysis_context = format_analysis_context(analysis_result, decision_text)

    # Add interpreter summary if provided (LLM output, already structured/validated upstream).
    interpretation_block = ""
    if interpretation:
        try:
            interpretation_block = "\nInterpreter Summary (post-simulation, no new numbers):\n" + json.dumps(
                interpretation, ensure_ascii=False
            ) + "\n"
        except Exception:
            interpretation_block = ""
    
    # Build conversation context
    safety = assess_safety((decision_text or "") + "\n" + (user_message or ""))

    conversation_context = """You are Regretless AI — a decision intelligence assistant.

You do NOT predict the future. You interpret Monte Carlo simulations.

Hard rules:
- Never invent numbers or probabilities.
- Never override the simulation results.
- Explain uncertainty in human terms.
- Ask clarifying questions only when confidence is low or context is missing.
- Challenge high-risk or irreversible decisions with preparation-first framing.

Guidelines:
- Reference specific numbers from the provided analysis when relevant (and only those).
- Be empathetic and practical.
- If the user mentions additional context (toxicity, stress, fear), acknowledge it and relate it to the analysis.
- If asked about outcomes, summarize best/most-likely/worst using the scenario data you have.

Tone: calm, neutral, supportive.
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
{interpretation_block}

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
        response = safe_chat(
            [
                {"role": "system", "content": "Return plain text. Do not add any numbers not already present."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=450,
        )
        reply = (response.text or "").strip()
        
        # Clean up response
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]

        # Safety prefix (high-risk domain or crisis)
        if safety.guidance_prefix:
            reply = safety.guidance_prefix + "\n\n" + reply

        # Guardrail: disallow invented numbers not present in context/prompt.
        allowed = set(extract_number_tokens(analysis_context))
        guarded, _extras = redact_invented_numbers(reply, allowed_numbers=allowed)
        return guarded.strip()
    
    except Exception:
        # Better error handling - try to give contextual response even on error
        from services.regret_calculator import get_regret_level, get_regret_explanation
        
        # Include user's message in a way that helps them understand we heard them
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ["toxic", "toxicity", "stress", "stressed", "worried", "concerned", "scared", "afraid"]):
            regret_level = get_regret_level(analysis_result.regret_score)
            return f"{safety.guidance_prefix + chr(10) + chr(10) if safety.guidance_prefix else ''}I hear you mentioning concerns about {user_message.lower()}. That's a really important factor in your decision.\n\nLooking at your analysis, your Regret Score is {analysis_result.regret_score:.2f} ({regret_level}), which suggests {get_regret_explanation(analysis_result.regret_score).lower()} Toxic environments and high stress can dominate long-term outcomes even when other metrics look okay.\n\nWould you like to share what a safer version of this decision would look like (e.g., a bridge plan or a short delay)?"
        else:
            # Still try to be helpful with context
            regret_level = get_regret_level(analysis_result.regret_score)
            return f"{safety.guidance_prefix + chr(10) + chr(10) if safety.guidance_prefix else ''}Based on your analysis, your Regret Score is {analysis_result.regret_score:.2f} ({regret_level}), which suggests {get_regret_explanation(analysis_result.regret_score).lower()}\n\nYou mentioned: '{user_message}'. What part of the decision feels most uncertain right now — money, stability, wellbeing, or relationships?"
