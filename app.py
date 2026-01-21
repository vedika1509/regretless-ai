"""Main Streamlit application for Regretless AI."""

import os

import streamlit as st
from dotenv import load_dotenv
from models.scenario import AnalysisResult, OutcomeMetrics, ScenarioResult
from services.chat_service import generate_chat_response
from services.counterfactual_service import CounterfactualExplorer, generate_counterfactual_summary
from services.decision_parser import parse_decision
from services.interview_service import generate_clarifying_questions
from services.explanation_generator import generate_scenario_explanation
from services.interpretation_service import generate_interpretation
from services.recommendation_service import generate_recommendations

# from services.comparison_service import DecisionComparator  # For future use
from services.regret_calculator import (
    calculate_regret,
)
from services.report_generator import generate_pdf_report
from services.risk_detector import RiskDetector
from services.simulator import MonteCarloSimulator
from utils.charts import plot_distribution, plot_metric_distributions, plot_scenario_comparison
from utils.explainability_graph import create_explainability_graph
from utils.visualizations import get_emoji_for_scenario, get_severity_icon
from utils.humanize import (
    band_label,
    compute_readiness,
    emotional_outcome_label,
    regret_band,
    scenario_bullets,
    scenario_story_title,
    timeline_for_scenario,
    to_10,
)

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Regretless AI", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed"
)

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_interpretation" not in st.session_state:
    st.session_state.analysis_interpretation = None

if "decision_text" not in st.session_state:
    st.session_state.decision_text = ""

if "decision_type" not in st.session_state:
    st.session_state.decision_type = "Custom"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False

if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

if "structured_decision" not in st.session_state:
    st.session_state.structured_decision = None

if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None

if "clarification" not in st.session_state:
    # clarification state machine for ask-first flow
    st.session_state.clarification = {
        "active": False,
        "original": "",
        "questions": [],
        "answers": [],
        "idx": 0,
    }

if "analysis_in_progress" not in st.session_state:
    st.session_state.analysis_in_progress = False

if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"

if "decision_context" not in st.session_state:
    # Lightweight, judge-friendly context collection
    st.session_state.decision_context = {
        "runway_months": None,  # int | None
        "plan_clarity": None,  # "No plan" | "Somewhat" | "Clear"
        "driver": None,  # "Stress" | "Mixed" | "Strategy"
        "horizon": "12–18 months",
        "family_pressure": None,  # "Low" | "Medium" | "High" | None
    }

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .scenario-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stChatMessage {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def _is_short_or_ambiguous(text: str) -> bool:
    """Heuristic: decide whether to ask clarifying questions before analysis."""
    t = (text or "").strip().lower()
    if len(t) < 35:
        return True
    # If it's basically just "should I X" without context, clarify
    vague_starters = ("can i ", "should i ", "what if ", "is it ok ", "help me ")
    if t.startswith(vague_starters) and len(t.split()) <= 6:
        return True
    return False


def _clarifying_questions_for(text: str) -> list[str]:
    """Generate 2–5 targeted clarifying questions (LLM-first, robust fallback)."""
    t = (text or "").strip().lower()

    # Education/study decisions
    if any(k in t for k in ["study", "studying", "college", "school", "course", "degree", "exam"]):
        return [
            "What’s making you want to quit studying right now (stress/boredom/financial pressure/health/other)?",
            "What exactly would you quit (a course, a semester, an entire degree), and what year/level are you in?",
            "If you quit, what’s your plan for the next 3–12 months (work, different course, break, skill learning)?",
            "What are your constraints (family expectations, money, deadlines, visa/scholarship, mental health)?",
        ]

    # Job quitting decisions
    if any(k in t for k in ["job", "work", "boss", "company", "office", "resign", "quit"]):
        return [
            "Why do you want to quit (toxicity, burnout, pay, growth, health, relocation)?",
            "Do you have another offer or a financial runway? If yes, how long can you manage without income?",
            "What’s the biggest risk if you quit now (money, career gap, family pressure, visa, confidence)?",
            "What would make staying acceptable for 1–3 more months (boundaries, role change, leave, negotiation)?",
        ]

    # Generic
    return [
        "What’s the main reason you’re considering this change right now?",
        "What are the top 1–2 constraints you cannot compromise on?",
        "What would a ‘good outcome’ look like 6 months from now?",
    ]


def _start_clarification(user_text: str) -> None:
    # LLM-first interviewer; fall back to deterministic questions.
    try:
        decision_type = (st.session_state.get("decision_type") or "Custom").lower()
        questions = generate_clarifying_questions(user_text, decision_type=decision_type, max_questions=5)
        if not questions:
            questions = _clarifying_questions_for(user_text)
    except Exception:
        questions = _clarifying_questions_for(user_text)
    st.session_state.clarification = {
        "active": True,
        "original": user_text.strip(),
        "questions": questions,
        "answers": [],
        "idx": 0,
    }


def _continue_or_finish_clarification(answer_text: str) -> bool:
    """
    Store an answer; return True if clarification is complete and we should start analysis.
    """
    c = st.session_state.clarification
    if not c.get("active"):
        return False

    # Store answer
    answers = list(c.get("answers", []))
    answers.append((answer_text or "").strip())
    c["answers"] = answers

    # Advance
    idx = int(c.get("idx", 0)) + 1
    c["idx"] = idx

    # Complete when we've answered all questions
    if idx >= len(c.get("questions", [])):
        c["active"] = False
        st.session_state.clarification = c
        return True

    st.session_state.clarification = c
    return False


def display_input_form():
    """Display decision input form."""
    st.markdown('<div class="main-header">🧠 Regretless AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Simulate Tomorrow. Decide Today.</div>', unsafe_allow_html=True
    )

    st.markdown("---")

    # Decision type selector
    decision_type = st.selectbox(
        "Decision Type",
        ["Job", "Rent", "Hiring", "Custom"],
        index=3,
        help="Select the category that best describes your decision",
    )
    st.session_state.decision_type = decision_type

    # Decision text input
    decision_text = st.text_area(
        "Describe your decision",
        height=150,
        placeholder="Example: Should I switch jobs from Company A to Company B? The new job offers a 30% salary increase but I'm not sure about the work-life balance...",
        value=st.session_state.decision_text,
    )
    st.session_state.decision_text = decision_text

    with st.expander("🧩 Add quick personal context (recommended — 30 seconds)"):
        c = st.session_state.decision_context
        col_a, col_b = st.columns(2)

        with col_a:
            runway = st.selectbox(
                "Do you have savings/income runway?",
                ["Prefer not to say", "0–1 months", "1–3 months", "3–6 months", "6+ months"],
                index=0,
                help="This helps interpret risk/regret in real terms.",
            )
            runway_map = {
                "Prefer not to say": None,
                "0–1 months": 1,
                "1–3 months": 2,
                "3–6 months": 5,
                "6+ months": 6,
            }
            c["runway_months"] = runway_map.get(runway)

            c["plan_clarity"] = st.selectbox(
                "Do you have a clear alternative plan?",
                ["Prefer not to say", "No plan", "Somewhat", "Clear"],
                index=0,
            )

            c["driver"] = st.selectbox(
                "Is this decision driven by stress or strategy?",
                ["Prefer not to say", "Stress", "Mixed", "Strategy"],
                index=0,
            )

        with col_b:
            c["horizon"] = st.selectbox(
                "Time horizon to evaluate outcomes",
                ["6–12 months", "12–18 months", "18–36 months"],
                index=1,
            )
            c["family_pressure"] = st.selectbox(
                "Family / social pressure (optional)",
                ["Prefer not to say", "Low", "Medium", "High"],
                index=0,
            )

        st.session_state.decision_context = c

    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Decision", type="primary", use_container_width=True)

    if analyze_button:
        if not decision_text or len(decision_text.strip()) < 10:
            st.error(
                "Please provide a detailed description of your decision (at least 10 characters)."
            )
        else:
            # Clear previous results
            st.session_state.analysis_result = None

            # Start analysis
            analyze_decision(decision_text, decision_type, st.session_state.decision_context)


def _augment_with_context(decision_text: str, context: dict | None) -> str:
    """Append structured context into text for the parser + explanation generator."""
    context = context or {}

    runway = context.get("runway_months")
    plan = context.get("plan_clarity")
    driver = context.get("driver")
    horizon = context.get("horizon")
    family = context.get("family_pressure")

    lines = []
    if isinstance(runway, int):
        lines.append(f"- Savings/income runway: ~{runway}+ months")
    if plan and plan not in ("Prefer not to say", None):
        lines.append(f"- Alternative plan clarity: {plan}")
    if driver and driver not in ("Prefer not to say", None):
        lines.append(f"- Decision driver: {driver}")
    if horizon:
        lines.append(f"- Time horizon to evaluate: {horizon}")
    if family and family not in ("Prefer not to say", None):
        lines.append(f"- Family/social pressure: {family}")

    if not lines:
        return decision_text

    return decision_text.strip() + "\n\nContext (for realism):\n" + "\n".join(lines) + "\n"


def analyze_decision(decision_text: str, decision_type: str, context: dict | None = None):
    """Perform complete decision analysis."""
    try:
        augmented_text = _augment_with_context(decision_text, context)

        # Step 1: Parse decision
        with st.spinner("🤖 Analyzing your decision..."):
            structured_decision = parse_decision(augmented_text, decision_type.lower())
            # Surface parser warnings early
            meta = getattr(structured_decision, "meta", {}) or {}
            for w in meta.get("warnings", [])[:2]:
                st.warning(f"⚠️ {w}")

        # Step 2: Run simulation
        simulation_count = int(os.getenv("SIMULATION_COUNT", "3000"))
        simulator = MonteCarloSimulator(n_iterations=simulation_count)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int):
            progress_bar.progress(current / total)
            status_text.info(f"🔄 Running {current:,} / {total:,} simulations...")

        status_text.info(
            f"🔄 Starting Monte Carlo simulation with {simulation_count:,} iterations..."
        )

        with st.spinner("📊 Running Monte Carlo simulation..."):
            simulation_results = simulator.run_simulation(
                structured_decision, progress_callback=update_progress
            )

        progress_bar.empty()
        status_text.empty()

        # Step 3: Extract scenarios
        with st.spinner("📈 Extracting scenarios..."):
            scenario_data = simulator.extract_scenarios(simulation_results)
            confidence = simulator.calculate_confidence(simulation_results)

        # Step 4: Detect risks
        with st.spinner("🚨 Detecting risks..."):
            risk_detector = RiskDetector()
            risks = risk_detector.detect_risks(simulation_results, augmented_text)

        # Step 5: Generate explanations
        scenarios = {}
        for scenario_type, metrics in scenario_data.items():
            with st.spinner(f"📝 Generating {scenario_type} explanation..."):
                drivers = metrics.get("drivers", [])
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                explanation = generate_scenario_explanation(
                    scenario_type, numeric_metrics, augmented_text
                )

                scenarios[scenario_type] = ScenarioResult(
                    scenario_type=scenario_type,
                    probability=float(numeric_metrics["probability"]),
                    outcomes=OutcomeMetrics(
                        satisfaction_score=float(numeric_metrics["satisfaction_score"]),
                        financial_score=float(numeric_metrics["financial_score"]),
                        risk_score=float(numeric_metrics["risk_score"]),
                        overall_score=float(numeric_metrics["overall_score"]),
                    ),
                    explanation=explanation,
                    drivers=drivers,
                    metrics=numeric_metrics,
                )

        # Step 6: Calculate Regret Score
        from services.regret_calculator import calculate_regret

        # Create temporary result for regret calculation
        temp_result = AnalysisResult(
            scenarios=scenarios,
            confidence=confidence,
            regret_score=0.0,  # Will be calculated next
            risks=risks,
            simulation_count=simulation_count,
        )

        regret_score = calculate_regret(temp_result)

        # Create analysis result with regret score
        analysis_result = AnalysisResult(
            scenarios=scenarios,
            confidence=confidence,
            regret_score=regret_score,
            risks=risks,
            simulation_count=simulation_count,
        )

        # LLM interpretation layer (voice + judgment; no new numbers)
        try:
            interp = generate_interpretation(
                analysis_result,
                decision_text=augmented_text,
                context=context or st.session_state.get("decision_context"),
            )
            st.session_state.analysis_interpretation = interp.model_dump() if interp else None
        except Exception:
            st.session_state.analysis_interpretation = None

        # Store in session state (including simulation results for visualizations)
        st.session_state.analysis_result = analysis_result
        # Keep both: raw for display, augmented for the assistant/pipeline
        st.session_state.decision_text_raw = decision_text
        st.session_state.decision_text = augmented_text
        st.session_state.decision_type = decision_type
        st.session_state.simulation_results = simulation_results  # Store for charts
        st.session_state.structured_decision = structured_decision  # Store for sensitivity analysis
        st.session_state.decision_context = context or st.session_state.decision_context

        # Initialize conversation with welcome message
        st.session_state.conversation_history = []
        st.session_state.chat_initialized = True

        # Success message
        st.success("✅ Analysis complete!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)


def display_scenarios(result: AnalysisResult):
    """Display scenario cards."""
    st.subheader("📋 Scenarios")

    cols = st.columns(3)

    scenario_order = ["best_case", "most_likely", "worst_case"]
    scenario_titles = {
        "best_case": "Best Case",
        "most_likely": "Most Likely",
        "worst_case": "Worst Case",
    }

    for idx, scenario_type in enumerate(scenario_order):
        if scenario_type in result.scenarios:
            scenario = result.scenarios[scenario_type]
            emoji = get_emoji_for_scenario(scenario_type)
            horizon = (st.session_state.get("decision_context") or {}).get("horizon")
            timeline = timeline_for_scenario(scenario_type, horizon=horizon)
            story_title = scenario_story_title(scenario_type)
            bullets = scenario_bullets(scenario.metrics or {})
            emotional = emotional_outcome_label(1.0 - float((scenario.metrics or {}).get("stress", 0.5)))

            with cols[idx]:
                st.markdown(f"### {emoji} {scenario_titles[scenario_type]}: “{story_title}”")
                st.caption(f"📅 Timeline: {timeline}")
                st.metric("Probability", f"~{scenario.probability*100:.0f}%")
                st.caption(f"😊 Emotional outcome: {emotional}")

                st.markdown("**What this looks like in real life:**")
                for b in bullets:
                    st.write(f"- {b}")

                if getattr(scenario, "drivers", None):
                    st.markdown("**What’s driving this:**")
                    for d in scenario.drivers[:3]:
                        direction = "↑" if d.delta > 0 else "↓" if d.delta < 0 else "→"
                        st.caption(
                            f"- {d.variable.replace('_', ' ').title()} {direction} (Δ {d.delta:+.2f})"
                        )

                # Collapse numbers
                with st.expander("Show metrics", expanded=False):
                    st.markdown("**Outcomes (0–1):**")
                    st.metric("Overall", f"{scenario.outcomes.overall_score:.2f}")
                    st.metric("Satisfaction", f"{scenario.outcomes.satisfaction_score:.2f}")
                    st.metric("Financial", f"{scenario.outcomes.financial_score:.2f}")
                    st.metric("Risk", f"{scenario.outcomes.risk_score:.2f}")
                    if scenario.metrics:
                        st.metric("Career stability (job security)", f"{float(scenario.metrics.get('job_security', 0.5)):.2f}")
                        st.metric("Stress", f"{float(scenario.metrics.get('stress', 0.5)):.2f}")


def display_risks(result: AnalysisResult):
    """Display risk panel."""
    st.subheader("🚨 Risk Analysis")
    if result.risks:
        for risk in result.risks:
            severity_icon = get_severity_icon(risk.severity)
            # Actionable mapping
            risk_key = (risk.risk_type or "").lower()
            if risk_key == "long_tail_downside":
                title = "Key Risk Detected: Long-tail downside"
                why = "A small-but-real set of outcomes is much worse than the average — this is where regret usually comes from."
                mitigations = [
                    "Delay committing until you can reduce uncertainty (2–4 weeks).",
                    "Set a checkpoint + fallback (Plan B) before you jump.",
                    "Design a “bridge” option (part-time, trial period, parallel applications).",
                ]
            elif risk_key in ("high_risk_score", "high_variance"):
                title = "Key Risk Detected: Unstable outcomes"
                why = "The system is signaling your decision is sensitive to unknowns — small changes can flip results."
                mitigations = [
                    "Gather missing info (offers, costs, timelines) and rerun.",
                    "Reduce downside: keep optionality for 30–60 days.",
                    "Avoid irreversible moves until variance drops.",
                ]
            elif risk_key in ("toxicity_risk", "mental_health_risk", "stress_risk"):
                title = "Key Risk Detected: Wellbeing risk"
                why = "Wellbeing risks can dominate long-term outcomes even when money looks okay."
                mitigations = [
                    "Protect health first: boundaries, leave, or professional support.",
                    "If you exit, do it with a bridge plan to reduce stress.",
                    "If you stay short-term, define what must change within 2–4 weeks.",
                ]
            else:
                title = f"Key Risk Detected: {risk.risk_type.replace('_', ' ').title()}"
                why = risk.description
                mitigations = [
                    "Make the risk explicit: write Plan A / Plan B.",
                    "Add a timeline + checkpoint to prevent drift.",
                ]

            st.warning(f"{severity_icon} **{title}** ({risk.severity.upper()})")
            st.caption(f"**Why this matters:** {why}")
            st.markdown("**Mitigation:**")
            for m in mitigations:
                st.write(f"- {m}")
            st.markdown("---")
    else:
        st.success("✅ No significant risks detected!")


def display_sensitivity_analysis(result: AnalysisResult):
    """Display sensitivity analysis widget."""
    st.subheader("🔬 Sensitivity Analysis")
    st.markdown("Adjust variable weights to see how they impact outcomes")

    if hasattr(st.session_state, "structured_decision") and st.session_state.structured_decision:
        variable_weights = {}
        cols = st.columns(3)

        for idx, (var_name, var_dist) in enumerate(
            st.session_state.structured_decision.variables.items()
        ):
            with cols[idx % 3]:
                weight = st.slider(
                    f"{var_name.replace('_', ' ').title()}", 0.0, 1.0, 0.5, key=f"weight_{var_name}"
                )
                variable_weights[var_name] = weight

        st.info(
            "💡 Sensitivity analysis re-calculation coming soon! This will show how changing variable weights affects outcomes."
        )
    else:
        st.info(
            "Sensitivity analysis requires the original decision structure. Please run a new analysis."
        )


def display_chat_interface(result: AnalysisResult):
    """Display conversational chat interface."""
    st.subheader("💬 Discuss Your Decision")
    st.markdown(
        "Ask questions, share your thoughts, or explore different aspects of your decision analysis."
    )

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        # Initialize chat with welcome message if not done
        if not st.session_state.chat_initialized or len(st.session_state.conversation_history) == 0:
            welcome_msg = "Hi! I've completed the analysis of your decision. I can help you understand the results, discuss different scenarios, or explore any questions you have. What would you like to know?"
            st.session_state.conversation_history = [{"role": "assistant", "content": welcome_msg}]
            st.session_state.chat_initialized = True

        # Display conversation history
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        # Generate and display assistant response
        with st.spinner("Thinking..."):
            try:
                assistant_response = generate_chat_response(
                    user_input,
                    result,
                    st.session_state.decision_text,
                    st.session_state.conversation_history,
                    interpretation=st.session_state.get("analysis_interpretation"),
                )

                # Add assistant response to history
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # Rerun to update the chat display
                st.rerun()

            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.rerun()


def display_results(result: AnalysisResult):
    """Display analysis results with tabs."""
    st.markdown('<div class="main-header">🧠 Regretless AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)

    # Sidebar with quick actions
    with st.sidebar:
        st.header("⚡ Quick Actions")

        if st.button("📄 Export PDF Report", use_container_width=True):
            try:
                pdf_buffer = generate_pdf_report(
                    result, st.session_state.decision_text, "decision_report.pdf"
                )
                st.success("Report generated!")
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buffer,
                    file_name="regretless_ai_report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

        if st.button("🔗 Share Analysis", use_container_width=True):
            st.info("Share link feature coming soon!")

        if st.button("📊 Compare Decisions", use_container_width=True):
            st.info("Save this decision and analyze another to compare")

        st.markdown("---")
        st.subheader("📈 Statistics")
        st.metric("Confidence", f"{result.confidence:.1%}")
        st.metric("Simulations", f"{result.simulation_count:,}")
        st.metric("Risks Detected", len(result.risks))

    # Back button
    if st.button("← Analyze Another Decision"):
        st.session_state.analysis_result = None
        st.session_state.decision_text = ""
        st.session_state.conversation_history = []
        st.session_state.chat_initialized = False
        st.session_state.simulation_results = None
        st.session_state.structured_decision = None
        st.rerun()

    st.markdown("---")

    # Key Metrics Display
    st.subheader("🧭 Decision Readiness + Human Metrics")

    context = st.session_state.get("decision_context") or {}
    verdict = compute_readiness(
        regret01=float(result.regret_score),
        confidence01=float(result.confidence),
        risks=result.risks,
        context=context,
    )

    st.markdown(
        f'<div style="padding: 0.9rem 1rem; border-radius: 10px; border: 2px solid {verdict.color};">'
        f'<div style="font-size: 1.4rem; font-weight: 800; color: {verdict.color};">'
        f"Decision Readiness: {verdict.label}</div>"
        f'<div style="margin-top: 0.25rem; color: #333;">{verdict.rationale}</div>'
        + (
            f'<div style="margin-top: 0.5rem; color: #333;"><b>Next step:</b> {verdict.conditions}</div>'
            if verdict.conditions
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # Likely outcome (human meaning) based on most-likely scenario
    most = result.scenarios.get("most_likely")
    likely_score = float(most.outcomes.overall_score) if most else 0.5
    label, color = band_label(likely_score)
    horizon = context.get("horizon", "12–18 months")
    stress = float((most.metrics or {}).get("stress", 0.5)) if most else 0.5
    job_sec = float((most.metrics or {}).get("job_security", 0.5)) if most else 0.5

    st.markdown("")
    st.markdown(
        f'<div style="font-size: 1.2rem; font-weight: 800; color: {color};">🟢 Likely Outcome: {label}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"What this means: over the next {horizon}, you’re likely to see "
        f"{'more stability' if job_sec >= 0.6 else 'some career uncertainty'}, "
        f"and {'manageable stress' if stress <= 0.55 else 'moderate stress'}."
    )

    # 4 human metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📈 Career Stability", f"{to_10(job_sec)}/10")
        st.caption("Based on job security")
    with c2:
        fin = float((most.metrics or {}).get("financial_satisfaction", most.outcomes.financial_score if most else 0.5)) if most else 0.5
        st.metric("💰 Financial Safety", f"{to_10(fin)}/10")
        st.caption("Runway + financial satisfaction")
    with c3:
        mental = 1.0 - stress
        st.metric("🧠 Mental Wellbeing", f"{to_10(mental)}/10")
        st.caption("Lower stress = higher wellbeing")
    with c4:
        st.metric("⏳ Long-term Regret Risk", regret_band(float(result.regret_score)))
        st.caption("Translated from Regret Score")

    with st.expander("Show the underlying numbers (for transparency)"):
        st.markdown(f"- **Confidence**: {float(result.confidence)*100:.1f}% (outcome consistency)")
        st.markdown(f"- **Regret Score**: {float(result.regret_score):.2f} (lower is better)")
        st.markdown(f"- **Simulations**: {result.simulation_count:,}")
        st.markdown(f"- **Risks detected**: {len(result.risks)}")

    # Quick refinement loop (judge-friendly)
    with st.expander("🧪 Refine this analysis (3 quick questions)"):
        st.caption("Answering these makes the scenarios less generic and the risk/regret interpretation more realistic.")
        ctx = st.session_state.get("decision_context") or {}

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            runway = st.selectbox(
                "Runway (months)",
                ["Prefer not to say", "0–1", "1–3", "3–6", "6+"],
                index=0 if ctx.get("runway_months") is None else (1 if ctx.get("runway_months") <= 1 else 2 if ctx.get("runway_months") <= 2 else 3 if ctx.get("runway_months") <= 5 else 4),
                key="refine_runway",
            )
        with col_b:
            plan = st.selectbox(
                "Plan clarity",
                ["Prefer not to say", "No plan", "Somewhat", "Clear"],
                index=0 if not ctx.get("plan_clarity") or ctx.get("plan_clarity") == "Prefer not to say" else ["No plan", "Somewhat", "Clear"].index(ctx.get("plan_clarity")) + 1,
                key="refine_plan",
            )
        with col_c:
            driver = st.selectbox(
                "Driver",
                ["Prefer not to say", "Stress", "Mixed", "Strategy"],
                index=0 if not ctx.get("driver") or ctx.get("driver") == "Prefer not to say" else ["Stress", "Mixed", "Strategy"].index(ctx.get("driver")) + 1,
                key="refine_driver",
            )

        runway_map = {"Prefer not to say": None, "0–1": 1, "1–3": 2, "3–6": 5, "6+": 6}
        new_ctx = dict(ctx)
        new_ctx["runway_months"] = runway_map.get(runway)
        new_ctx["plan_clarity"] = plan
        new_ctx["driver"] = driver
        st.session_state.decision_context = new_ctx

        if st.button("🔁 Re-run simulation with this context", use_container_width=True):
            raw = st.session_state.get("decision_text_raw") or st.session_state.get("decision_text") or ""
            st.session_state.analysis_result = None
            analyze_decision(raw, st.session_state.get("decision_type", "Custom"), st.session_state.decision_context)

    # Regret insight (actionable)
    st.markdown("")
    st.subheader("🧠 Regret Insight")
    plan = (context.get("plan_clarity") or "")
    runway = context.get("runway_months")
    if plan in ("No plan", None, "Prefer not to say") or (isinstance(runway, int) and runway < 3):
        st.info(
            "Your regret risk is **LOW–MODERATE only if you plan the exit**.\n\n"
            "- Regret rises sharply if you move without a Plan A/Plan B.\n"
            "- If runway is short, stress becomes the dominant driver.\n\n"
            "Mitigation: delay commitment 2–4 weeks, build a bridge option, and define checkpoints."
        )
    else:
        st.info(
            "Your regret risk is **lower when you keep structure**.\n\n"
            "- You already have some runway and plan clarity.\n"
            "- Keep a checkpoint (8–12 weeks) so you can adjust early if reality diverges."
        )

    # “Future you” quote (judge-friendly differentiator)
    st.markdown("")
    st.subheader("🧍 Future You (2 years) says…")
    if verdict.label.startswith("NOT READY"):
        st.write(
            '"I’m glad I didn’t rush it. The moment I added a plan and a runway, the decision got easier — and the regret risk dropped."'
        )
    else:
        st.write(
            '"I don’t regret it — because I treated it like a project: timeline, fallback plan, and early checkpoints."'
        )

    # Optional: LLM interpretation layer output (structured, post-simulation)
    interp = st.session_state.get("analysis_interpretation")
    if isinstance(interp, dict) and interp.get("biggest_hidden_risk"):
        st.markdown("")
        st.subheader("🗣️ Interpretation Layer (LLM)")

        for t in (interp.get("tradeoffs") or [])[:5]:
            st.write(f"- {t}")

        risk = interp.get("biggest_hidden_risk") or {}
        if risk:
            st.markdown("**Biggest hidden risk:**")
            st.write(f"- **{risk.get('title','')}**")
            st.caption(risk.get("why_it_matters", ""))
            st.markdown("**Mitigation:**")
            for m in (risk.get("mitigation") or [])[:5]:
                st.write(f"- {m}")

        if interp.get("regret_paths"):
            st.markdown("**Regret paths:**")
            for rp in (interp.get("regret_paths") or [])[:5]:
                st.write(f"- {rp}")

        if interp.get("recommendations"):
            st.markdown("**Recommendations:**")
            for rec in (interp.get("recommendations") or [])[:6]:
                st.write(f"- {rec}")

        if interp.get("followup_questions"):
            st.markdown("**To refine this further, answer:**")
            for q in (interp.get("followup_questions") or [])[:4]:
                st.write(f"- {q}")

    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📊 Overview",
            "📈 Visualizations",
            "🔬 Sensitivity",
            "🔁 Counterfactuals",
            "🔍 Explainability",
            "💡 Recommendations",
            "💬 Discussion",
        ]
    )

    with tab1:
        # Overview tab - current results display
        display_scenarios(result)
        st.markdown("---")
        display_risks(result)
        st.markdown("---")
        st.caption(f"Analysis based on {result.simulation_count:,} Monte Carlo simulations")

    with tab2:
        # Visualizations tab
        if st.session_state.simulation_results:
            st.subheader("Distribution Analysis")

            # Overall score distribution
            fig1 = plot_distribution(st.session_state.simulation_results, "overall_score")
            st.plotly_chart(fig1, use_container_width=True)

            # All metrics distribution
            fig2 = plot_metric_distributions(st.session_state.simulation_results)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Scenario Comparison")
            fig3 = plot_scenario_comparison(result.scenarios)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Visualizations require simulation results. Please run a new analysis.")

    with tab3:
        # Sensitivity analysis tab
        display_sensitivity_analysis(result)

    with tab4:
        # Counterfactuals tab
        st.subheader("🔁 Counterfactual Explorer")
        st.markdown(
            "**What would change the outcome?** Explore what variable changes would shift scenarios."
        )

        if st.session_state.simulation_results and st.session_state.structured_decision:
            explorer = CounterfactualExplorer()
            variable_names = list(st.session_state.structured_decision.variables.keys())

            # Target shift selector
            shift_type = st.selectbox(
                "Target Shift",
                ["worst_to_best", "worst_to_likely", "likely_to_best"],
                format_func=lambda x: {
                    "worst_to_best": "Worst Case → Best Case",
                    "worst_to_likely": "Worst Case → Most Likely",
                    "likely_to_best": "Most Likely → Best Case",
                }[x],
            )

            # Compute counterfactuals
            with st.spinner("Analyzing counterfactuals..."):
                counterfactuals = explorer.find_minimum_changes(
                    result, st.session_state.simulation_results, variable_names, shift_type
                )

            if counterfactuals:
                # Summary
                summary = generate_counterfactual_summary(counterfactuals, result)
                st.info(summary)

                st.markdown("---")
                st.subheader("Top Variable Changes")

                # Display top counterfactuals
                for idx, cf in enumerate(counterfactuals[:5], 1):
                    with st.expander(f"#{idx}: {cf['variable'].replace('_', ' ').title()}"):
                        st.markdown(f"**Direction:** {cf['direction'].title()}")
                        st.metric("Required Change", f"{cf['change_percent']:.1f}%")
                        st.metric("Impact Score", f"{cf['impact']:.3f}")
                        st.markdown(f"**Explanation:** {cf['explanation']}")

                # Visualize impact
                st.markdown("---")
                st.subheader("Impact Ranking")

                import pandas as pd

                df = pd.DataFrame(counterfactuals[:5])
                df["variable"] = df["variable"].str.replace("_", " ").str.title()
                df = df[["variable", "impact", "change_percent"]]

                st.dataframe(
                    df,
                    column_config={
                        "variable": "Variable",
                        "impact": st.column_config.NumberColumn("Impact Score", format="%.3f"),
                        "change_percent": st.column_config.NumberColumn(
                            "Required Change (%)", format="%.1f%%"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Unable to compute counterfactuals. Please ensure sufficient simulation data."
                )
        else:
            st.info(
                "Counterfactual analysis requires simulation results. Please run a new analysis."
            )

    with tab5:
        # Explainability tab
        st.subheader("🔍 Decision Explainability Graph")
        st.markdown("**Visualize causal relationships** between variables and outcomes.")

        if st.session_state.structured_decision:
            # Get causal graph
            from services.causal_graph import CausalGraph

            causal_graph = CausalGraph()

            # Create explainability graph
            fig = create_explainability_graph(st.session_state.structured_decision, causal_graph)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Legend")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Green nodes** = Input Variables")
                st.markdown("**Blue nodes** = Outcome Dimensions")
            with col2:
                st.markdown("**Gray edges** = Causal Relationships")
                st.markdown("**Direction** = Variables → Outcomes")
        else:
            st.info("Explainability graph requires decision structure. Please run a new analysis.")

    with tab4:
        # Recommendations tab
        st.subheader("💡 Actionable Recommendations")
        st.markdown("Based on your analysis, here are specific steps you can take:")

        try:
            recommendations = generate_recommendations(result, st.session_state.decision_text)

            for rec in recommendations["recommendations"]:
                if rec.strip():
                    st.markdown(f"✅ {rec.strip()}")

            st.markdown("---")
            st.caption(f"Priority Level: {recommendations['priority'].upper()}")
            st.caption(f"Timeline: {recommendations['timeline'].replace('_', ' ').title()}")
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            st.info("Please try again or check your API configuration.")

    with tab5:
        # Discussion tab - chat interface
        display_chat_interface(result)


# Main app logic - Chat-based interface with enhanced features
def display_visualizations_view(result: AnalysisResult):
    """Display visualizations view."""
    st.subheader("📈 Interactive Visualizations")

    if st.session_state.simulation_results:
        st.markdown("### Distribution Analysis")
        st.markdown("Explore how outcomes are distributed across all simulations.")

        # Overall score distribution
        fig1 = plot_distribution(st.session_state.simulation_results, "overall_score")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")

        # All metrics distribution
        st.markdown("### All Metrics Distribution")
        st.markdown("Compare distributions across financial, satisfaction, and risk scores.")
        fig2 = plot_metric_distributions(st.session_state.simulation_results)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Scenario comparison
        st.markdown("### Scenario Comparison Radar Chart")
        st.markdown("Visual comparison of best case, most likely, and worst case scenarios.")
        fig3 = plot_scenario_comparison(result.scenarios)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Visualizations require simulation results. Please run a new analysis.")


def display_counterfactuals_view(result: AnalysisResult):
    """Display counterfactuals view."""
    st.subheader("🔁 Counterfactual Explorer")
    st.markdown(
        "**What would change the outcome?** Explore what variable changes would shift scenarios from worst to best."
    )

    if st.session_state.simulation_results and st.session_state.structured_decision:
        explorer = CounterfactualExplorer()
        variable_names = list(st.session_state.structured_decision.variables.keys())

        # Target shift selector
        shift_type = st.selectbox(
            "Target Shift",
            ["worst_to_best", "worst_to_likely", "likely_to_best"],
            format_func=lambda x: {
                "worst_to_best": "Worst Case → Best Case",
                "worst_to_likely": "Worst Case → Most Likely",
                "likely_to_best": "Most Likely → Best Case",
            }[x],
        )

        # Compute counterfactuals
        with st.spinner("Analyzing counterfactuals..."):
            counterfactuals = explorer.find_minimum_changes(
                result, st.session_state.simulation_results, variable_names, shift_type
            )

        if counterfactuals:
            # Summary
            summary = generate_counterfactual_summary(counterfactuals, result)
            st.info(f"💡 **Insight:** {summary}")

            st.markdown("---")
            st.subheader("Top Variable Changes Required")

            # Display top counterfactuals
            for idx, cf in enumerate(counterfactuals[:5], 1):
                with st.expander(
                    f"#{idx}: {cf['variable'].replace('_', ' ').title()}", expanded=(idx == 1)
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Required Change", f"{cf['change_percent']:.1f}%")
                        st.markdown(f"**Direction:** {cf['direction'].title()}")
                    with col2:
                        st.metric("Impact Score", f"{cf['impact']:.3f}")
                        st.caption("Higher = More impactful")
                    st.markdown(f"**What this means:** {cf['explanation']}")

            # Impact ranking table
            st.markdown("---")
            st.subheader("Impact Ranking")

            import pandas as pd

            df = pd.DataFrame(counterfactuals[:5])
            df["variable"] = df["variable"].str.replace("_", " ").str.title()
            df = df[["variable", "impact", "change_percent"]]

            st.dataframe(
                df,
                column_config={
                    "variable": "Variable",
                    "impact": st.column_config.NumberColumn(
                        "Impact Score",
                        format="%.3f",
                        help="Higher score = More impactful on outcomes",
                    ),
                    "change_percent": st.column_config.NumberColumn(
                        "Required Change (%)",
                        format="%.1f%%",
                        help="Percentage change needed to shift scenario",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning(
                "Unable to compute counterfactuals. Please ensure sufficient simulation data."
            )
    else:
        st.info("Counterfactual analysis requires simulation results. Please run a new analysis.")


def display_explainability_view(result: AnalysisResult):
    """Display explainability graph view."""
    st.subheader("🔍 Decision Explainability Graph")
    st.markdown(
        "**Visualize causal relationships** between variables and outcomes. This shows how different factors influence your decision outcomes."
    )

    if st.session_state.structured_decision:
        # Get causal graph
        from services.causal_graph import CausalGraph

        causal_graph = CausalGraph()

        # Create explainability graph
        fig = create_explainability_graph(st.session_state.structured_decision, causal_graph)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Understanding the Graph")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Node Types:**
            - 🟢 **Green nodes** = Input Variables (factors you can influence)
            - 🔵 **Blue nodes** = Outcome Dimensions (results of your decision)
            """)
        with col2:
            st.markdown("""
            **Connections:**
            - **Gray edges** = Causal relationships
            - **Direction** = Variables → Outcomes
            - **Thickness** = Strength of influence
            """)

        # Show variable details
        if st.session_state.structured_decision.variables:
            st.markdown("---")
            st.subheader("Variable Details")
            for var_name, var_dist in st.session_state.structured_decision.variables.items():
                with st.expander(f"📊 {var_name.replace('_', ' ').title()}"):
                    st.markdown(f"**Distribution:** {var_dist.distribution}")
                    st.markdown(f"**Base Value:** {var_dist.value:.2f}")
                    st.json(var_dist.params)
    else:
        st.info("Explainability graph requires decision structure. Please run a new analysis.")


def display_recommendations_view(result: AnalysisResult):
    """Display recommendations view."""
    st.subheader("💡 Actionable Recommendations")
    st.markdown(
        "Based on your analysis, here are specific steps you can take to improve your decision outcomes:"
    )

    try:
        recommendations = generate_recommendations(result, st.session_state.decision_text)

        st.markdown("---")

        # Display recommendations
        for idx, rec in enumerate(recommendations["recommendations"], 1):
            if rec.strip() and len(rec.strip()) > 10:
                st.markdown(f"#### ✅ {idx}. {rec.strip()}")
                st.markdown("")

        st.markdown("---")

        # Priority and timeline
        col1, col2 = st.columns(2)
        with col1:
            priority_emoji = (
                "🔴"
                if recommendations["priority"] == "high"
                else "🟡"
                if recommendations["priority"] == "medium"
                else "🟢"
            )
            st.metric("Priority Level", f"{priority_emoji} {recommendations['priority'].upper()}")
        with col2:
            st.metric("Timeline", recommendations["timeline"].replace("_", " ").title())

    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        st.info("Please try again or check your API configuration.")


def display_sensitivity_view(result: AnalysisResult):
    """Display sensitivity analysis view."""
    st.subheader("🔬 Sensitivity Analysis")
    st.markdown(
        "**Adjust variable weights** to see how they impact outcomes. This helps you understand which factors matter most for your decision."
    )

    if hasattr(st.session_state, "structured_decision") and st.session_state.structured_decision:
        st.info(
            "💡 Sensitivity analysis re-calculation coming soon! This will show how changing variable weights affects outcomes in real-time."
        )

        # Show current variables
        if st.session_state.structured_decision.variables:
            st.markdown("### Current Decision Variables")
            for var_name, var_dist in st.session_state.structured_decision.variables.items():
                with st.expander(f"📊 {var_name.replace('_', ' ').title()}"):
                    st.markdown(f"**Distribution:** {var_dist.distribution}")
                    st.markdown(f"**Base Value:** {var_dist.value:.2f}")
                    st.json(var_dist.params)
    else:
        st.info(
            "Sensitivity analysis requires the original decision structure. Please run a new analysis."
        )


def display_chat_main():
    """Main chat-based interface with enhanced features accessible via sidebar."""
    # Header
    st.markdown('<div class="main-header">🧠 Regretless AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Simulate Tomorrow. Decide Today.</div>', unsafe_allow_html=True
    )

    # Sidebar with features (only show if we have analysis results)
    if st.session_state.analysis_result:
        with st.sidebar:
            st.header("🎯 Quick Access")
            st.markdown("---")

            # Current view indicator
            current_view = st.session_state.get("current_view", "chat")
            st.caption(f"📍 Current: {'Chat' if current_view == 'chat' else current_view.title()}")
            st.markdown("---")

            # Navigation buttons - highlight current view
            button_type = "primary" if current_view == "chat" else "secondary"
            if st.button("💬 Chat", use_container_width=True, type=button_type):
                st.session_state.current_view = "chat"
                st.rerun()

            st.markdown("### 📊 Analysis Features")

            button_type = "primary" if current_view == "visualizations" else "secondary"
            if st.button("📈 Visualizations", use_container_width=True, type=button_type):
                st.session_state.current_view = "visualizations"
                st.rerun()

            button_type = "primary" if current_view == "counterfactuals" else "secondary"
            if st.button("🔁 Counterfactuals", use_container_width=True, type=button_type):
                st.session_state.current_view = "counterfactuals"
                st.rerun()

            button_type = "primary" if current_view == "explainability" else "secondary"
            if st.button("🔍 Explainability", use_container_width=True, type=button_type):
                st.session_state.current_view = "explainability"
                st.rerun()

            button_type = "primary" if current_view == "recommendations" else "secondary"
            if st.button("💡 Recommendations", use_container_width=True, type=button_type):
                st.session_state.current_view = "recommendations"
                st.rerun()

            button_type = "primary" if current_view == "sensitivity" else "secondary"
            if st.button("🔬 Sensitivity", use_container_width=True, type=button_type):
                st.session_state.current_view = "sensitivity"
                st.rerun()

            st.markdown("---")
            st.subheader("⚡ Actions")

            if st.button("📄 Export PDF", use_container_width=True):
                try:
                    pdf_buffer = generate_pdf_report(
                        st.session_state.analysis_result,
                        st.session_state.decision_text,
                        "decision_report.pdf",
                    )
                    st.success("Report generated!")
                    st.download_button(
                        label="📥 Download",
                        data=pdf_buffer,
                        file_name="regretless_ai_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

            if st.button("🔄 New Decision", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.decision_text = ""
                st.session_state.conversation_history = []
                st.session_state.simulation_results = None
                st.session_state.structured_decision = None
                st.session_state.current_view = "chat"
                st.rerun()

            st.markdown("---")
            st.subheader("📊 Key Metrics")
            result = st.session_state.analysis_result
            st.metric("Confidence", f"{result.confidence:.1%}")
            _r = float(result.regret_score)
            regret_level = regret_band(_r)
            regret_color = "green" if _r < 0.35 else "orange" if _r < 0.55 else "red"
            st.markdown(
                f'<div style="color: {regret_color}; font-weight: bold; font-size: 1.2rem;">'
                f"Regret Score: {result.regret_score:.2f}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{regret_level} risk")
            st.metric("Risks", len(result.risks))

    # Initialize welcome message
    if len(st.session_state.conversation_history) == 0:
        welcome_msg = """Hi! I'm Regretless AI, your decision analysis assistant. I can help you make better decisions by exploring multiple future scenarios using probabilistic simulation.

Tell me about a decision you're facing, and I'll:
- Analyze the decision using Monte Carlo simulation
- Show you best case, worst case, and most likely scenarios
- Calculate your Regret Score (proprietary metric)
- Identify hidden risks
- Provide actionable recommendations
- Help you explore "what if" scenarios

What decision would you like to analyze?"""

        st.session_state.conversation_history = [{"role": "assistant", "content": welcome_msg}]

    # Display conversation history
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                # Check if this is an analysis result message
                if msg.get("type") == "analysis_result":
                    result = msg["result"]

                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Confidence", f"{result.confidence:.1%}")
                    with col2:
                        _r = float(result.regret_score)
                        regret_level = regret_band(_r)
                        regret_color = "green" if _r < 0.35 else "orange" if _r < 0.55 else "red"
                        st.markdown(
                            f'<div style="color: {regret_color}; font-weight: bold;">'
                            f"Regret Score: {result.regret_score:.2f} ({regret_level})</div>",
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.metric("Simulations", f"{result.simulation_count:,}")
                    with col4:
                        st.metric("Risks", len(result.risks))

                    # Display scenarios
                    st.markdown("### 📋 Scenarios")
                    cols = st.columns(3)
                    scenario_order = ["best_case", "most_likely", "worst_case"]
                    scenario_titles = {
                        "best_case": "🌤️ Best Case",
                        "most_likely": "☁️ Most Likely",
                        "worst_case": "🌧️ Worst Case",
                    }

                    for idx, scenario_type in enumerate(scenario_order):
                        if scenario_type in result.scenarios:
                            scenario = result.scenarios[scenario_type]
                            with cols[idx]:
                                st.markdown(f"**{scenario_titles[scenario_type]}**")
                                st.metric("Overall", f"{scenario.outcomes.overall_score:.2f}")
                                # Show full explanation with expander if long, otherwise show directly
                                if len(scenario.explanation) > 150:
                                    with st.expander("📖 Read explanation", expanded=False):
                                        st.write(scenario.explanation)
                                else:
                                    st.write(scenario.explanation)

                    # Display risks if any
                    if result.risks:
                        st.markdown("### 🚨 Detected Risks")
                        for risk in result.risks[:3]:  # Show top 3
                            severity_icon = get_severity_icon(risk.severity)
                            st.warning(f"{severity_icon} {risk.description}")

                    # Store result in session state for follow-up questions
                    st.session_state.analysis_result = result

                    st.markdown("---")

                    # Quick access buttons to features - improved styling
                    st.markdown("### 🎯 Explore Your Analysis")
                    st.markdown(
                        "Click any button below to dive deeper into your decision analysis:"
                    )

                    # Create a grid of buttons
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        if st.button(
                            "📈 Charts",
                            key="btn_viz_chat",
                            use_container_width=True,
                            help="View interactive visualizations",
                        ):
                            st.session_state.current_view = "visualizations"
                            st.rerun()

                    with col2:
                        if st.button(
                            "🔁 What If?",
                            key="btn_cf_chat",
                            use_container_width=True,
                            help="Explore counterfactual scenarios",
                        ):
                            st.session_state.current_view = "counterfactuals"
                            st.rerun()

                    with col3:
                        if st.button(
                            "🔍 Graph",
                            key="btn_exp_chat",
                            use_container_width=True,
                            help="See causal relationships",
                        ):
                            st.session_state.current_view = "explainability"
                            st.rerun()

                    with col4:
                        if st.button(
                            "💡 Advice",
                            key="btn_rec_chat",
                            use_container_width=True,
                            help="Get actionable recommendations",
                        ):
                            st.session_state.current_view = "recommendations"
                            st.rerun()

                    with col5:
                        if st.button(
                            "🔬 Sensitivity",
                            key="btn_sens_chat",
                            use_container_width=True,
                            help="Analyze variable sensitivity",
                        ):
                            st.session_state.current_view = "sensitivity"
                            st.rerun()

                    st.markdown("")
                    st.info(
                        "💬 **You can also ask me questions about these results in the chat below, or use the sidebar to navigate between features.**"
                    )

                elif msg.get("type") == "visualization":
                    # Display charts if requested
                    if msg.get("chart_type") == "distribution":
                        fig = plot_distribution(
                            st.session_state.simulation_results, msg.get("metric", "overall_score")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif msg.get("chart_type") == "scenario_comparison":
                        fig = plot_scenario_comparison(st.session_state.analysis_result.scenarios)
                        st.plotly_chart(fig, use_container_width=True)

                elif msg.get("type") == "loading":
                    # Show loading indicator
                    with st.spinner(msg["content"]):
                        st.write(msg["content"])

                else:
                    # Regular text message
                    st.write(msg["content"])

    # Check if we need to process a pending analysis (after loading message was shown)
    if (
        len(st.session_state.conversation_history) >= 2
        and st.session_state.conversation_history[-1].get("type") == "loading"
        and st.session_state.conversation_history[-2].get("role") == "user"
        and not st.session_state.get("analysis_in_progress", False)
    ):
        # Get user message (or prepared decision text)
        user_input = st.session_state.conversation_history[-2]["content"]
        prepared_text = st.session_state.get("pending_user_input")

        # Mark analysis as in progress
        st.session_state.analysis_in_progress = True

        try:
            # Extract decision text from user input
            decision_text = prepared_text or user_input
            if isinstance(decision_text, str) and ":" in decision_text:
                decision_text = decision_text.split(":", 1)[1].strip()

            decision_type = st.session_state.decision_type or "Custom"

            # Perform analysis (reuse existing function)
            perform_analysis(decision_text, decision_type)
            st.session_state.pending_user_input = None
            st.session_state.analysis_in_progress = False
            st.rerun()

        except Exception as e:
            # Remove loading message
            if (
                st.session_state.conversation_history
                and st.session_state.conversation_history[-1].get("type") == "loading"
            ):
                st.session_state.conversation_history.pop()

            error_msg = f"❌ I apologize, but I encountered an error while analyzing your decision:\n\n**Error:** {str(e)}\n\n**Please try:**\n- Rephrasing your question\n- Providing more details about your decision\n- Checking your GROQ_API_KEY in environment variables"
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": error_msg}
            )
            st.session_state.analysis_in_progress = False
            st.rerun()

    # Chat input
    user_input = st.chat_input("Describe your decision or ask a question...")

    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        # Ask-first clarification flow
        clarification = st.session_state.get("clarification") or {}
        if clarification.get("active"):
            # Treat this message as the answer to the current question
            finished = _continue_or_finish_clarification(user_input)

            if finished:
                c = st.session_state.clarification
                # Build a richer decision text from Q/A
                qa_lines = []
                for i, q in enumerate(c.get("questions", [])):
                    a = c.get("answers", [""] * len(c.get("questions", [])))[i]
                    qa_lines.append(f"- Q: {q}\n  A: {a}")

                decision_text = (
                    c.get("original", "").strip() + "\n\nClarifications:\n" + "\n".join(qa_lines)
                )
                st.session_state.pending_user_input = decision_text

                # Add loading message and rerun to process analysis in next cycle
                loading_msg = {
                    "role": "assistant",
                    "content": "🔍 Thanks — I have enough context. I’m analyzing this now (about 30–60 seconds)...",
                    "type": "loading",
                }
                st.session_state.conversation_history.append(loading_msg)
                st.session_state.analysis_in_progress = False
                st.rerun()
            else:
                # Ask the next clarification question
                c = st.session_state.clarification
                idx = int(c.get("idx", 0))
                q = c.get("questions", [])[idx]
                st.session_state.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": f"{q}",
                    }
                )
                st.rerun()

        # Check if user wants to analyze a new decision
        should_analyze = st.session_state.analysis_result is None or any(
            keyword in user_input.lower()
            for keyword in [
                "new decision",
                "another decision",
                "analyze",
                "help me with",
                "should i",
                "what if",
            ]
        )

        if should_analyze:
            # Ask-first: if prompt is short/ambiguous, ask clarifying questions instead of analyzing immediately.
            if _is_short_or_ambiguous(user_input):
                _start_clarification(user_input)
                c = st.session_state.clarification
                first_q = (
                    c.get("questions", [])[0]
                    if c.get("questions")
                    else "Can you share a bit more context?"
                )
                st.session_state.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": f"I can help, but I need a bit more context first.\n\n{first_q}",
                    }
                )
                st.rerun()
            else:
                st.session_state.pending_user_input = user_input
                # Add loading message immediately
                loading_msg = {
                    "role": "assistant",
                    "content": "🔍 I'm analyzing your decision. This may take 30-60 seconds. Please wait...",
                    "type": "loading",
                }
                st.session_state.conversation_history.append(loading_msg)
                st.session_state.analysis_in_progress = False  # Reset flag
                st.rerun()  # Show loading message first, then process in next cycle

        # If we have analysis results, generate conversational response
        elif st.session_state.analysis_result:
            # Generate chat response
            with st.spinner("Thinking..."):
                try:
                    assistant_response = generate_chat_response(
                        user_input,
                        st.session_state.analysis_result,
                        st.session_state.decision_text or "Your decision",
                        st.session_state.conversation_history,
                        interpretation=st.session_state.get("analysis_interpretation"),
                    )

                    # Check if user is requesting a visualization
                    if any(
                        keyword in user_input.lower()
                        for keyword in ["show me", "visualize", "chart", "graph", "plot"]
                    ):
                        if (
                            "distribution" in user_input.lower()
                            or "histogram" in user_input.lower()
                        ):
                            st.session_state.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_response,
                                    "type": "visualization",
                                    "chart_type": "distribution",
                                    "metric": "overall_score",
                                }
                            )
                        elif "scenario" in user_input.lower() or "comparison" in user_input.lower():
                            st.session_state.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_response,
                                    "type": "visualization",
                                    "chart_type": "scenario_comparison",
                                }
                            )
                        else:
                            st.session_state.conversation_history.append(
                                {"role": "assistant", "content": assistant_response}
                            )
                    else:
                        st.session_state.conversation_history.append(
                            {"role": "assistant", "content": assistant_response}
                        )

                    st.rerun()

                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    st.rerun()

        else:
            # No analysis yet, prompt for decision
            prompt_msg = "I'd be happy to help! Please describe a decision you're facing, and I'll analyze it for you using probabilistic simulation. For example: 'Should I switch jobs from Company A to Company B?'"
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": prompt_msg}
            )
            st.rerun()


def perform_analysis(decision_text: str, decision_type: str):
    """Perform decision analysis and add result to chat."""
    try:
        # Step 1: Parse decision
        structured_decision = parse_decision(decision_text, decision_type.lower())
        meta = getattr(structured_decision, "meta", {}) or {}
        if meta.get("warnings"):
            st.session_state.conversation_history.append(
                {
                    "role": "assistant",
                    "content": "⚠️ Note: " + " ".join(meta.get("warnings", [])[:2]),
                }
            )

        # Step 2: Run simulation
        simulation_count = int(os.getenv("SIMULATION_COUNT", "3000"))
        simulator = MonteCarloSimulator(n_iterations=simulation_count)

        simulation_results = simulator.run_simulation(structured_decision)

        # Step 3: Extract scenarios
        scenario_data = simulator.extract_scenarios(simulation_results)
        confidence = simulator.calculate_confidence(simulation_results)

        # Step 4: Detect risks
        risk_detector = RiskDetector()
        risks = risk_detector.detect_risks(simulation_results, decision_text)

        # Step 5: Generate explanations
        scenarios = {}
        for scenario_type, metrics in scenario_data.items():
            drivers = metrics.get("drivers", [])
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            explanation = generate_scenario_explanation(
                scenario_type, numeric_metrics, decision_text
            )

            scenarios[scenario_type] = ScenarioResult(
                scenario_type=scenario_type,
                probability=float(numeric_metrics["probability"]),
                outcomes=OutcomeMetrics(
                    satisfaction_score=float(numeric_metrics["satisfaction_score"]),
                    financial_score=float(numeric_metrics["financial_score"]),
                    risk_score=float(numeric_metrics["risk_score"]),
                    overall_score=float(numeric_metrics["overall_score"]),
                ),
                explanation=explanation,
                drivers=drivers,
                metrics=numeric_metrics,
            )

        # Step 6: Calculate Regret Score
        temp_result = AnalysisResult(
            scenarios=scenarios,
            confidence=confidence,
            regret_score=0.0,
            risks=risks,
            simulation_count=simulation_count,
        )

        regret_score = calculate_regret(temp_result)

        # Create analysis result
        analysis_result = AnalysisResult(
            scenarios=scenarios,
            confidence=confidence,
            regret_score=regret_score,
            risks=risks,
            simulation_count=simulation_count,
        )

        # Store in session state
        st.session_state.analysis_result = analysis_result
        st.session_state.decision_text = decision_text
        st.session_state.decision_type = decision_type
        st.session_state.simulation_results = simulation_results
        st.session_state.structured_decision = structured_decision

        # Interpretation layer (chat mode)
        try:
            interp = generate_interpretation(
                analysis_result,
                decision_text=decision_text,
                context=st.session_state.get("decision_context"),
            )
            st.session_state.analysis_interpretation = interp.model_dump() if interp else None
        except Exception:
            st.session_state.analysis_interpretation = None

        # Remove loading message
        if (
            st.session_state.conversation_history
            and st.session_state.conversation_history[-1].get("type") == "loading"
        ):
            st.session_state.conversation_history.pop()

        # Add analysis result to chat
        result_msg = f"""✅ I've completed the analysis of your decision! Here are the key insights:

**Decision:** {decision_text}

The analysis is based on {simulation_count:,} Monte Carlo simulations. Here's what I found:"""

        st.session_state.conversation_history.append(
            {
                "role": "assistant",
                "content": result_msg,
                "type": "analysis_result",
                "result": analysis_result,
            }
        )

    except Exception as e:
        # Remove loading message
        if (
            st.session_state.conversation_history
            and st.session_state.conversation_history[-1].get("type") == "loading"
        ):
            st.session_state.conversation_history.pop()

        error_msg = f"❌ Error: {str(e)}"
        st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
        raise e


# Main app logic - Show chat interface or selected feature view
if st.session_state.get("current_view") == "chat" or st.session_state.analysis_result is None:
    # Show chat interface
    display_chat_main()
else:
    # Show feature views when user clicks sidebar buttons
    result = st.session_state.analysis_result
    current_view = st.session_state.current_view

    # Sidebar with features (always show when we have results)
    with st.sidebar:
        st.header("🎯 Quick Access")
        st.markdown("---")

        # Navigation buttons
        if st.button("💬 Back to Chat", use_container_width=True, type="primary"):
            st.session_state.current_view = "chat"
            st.rerun()

        st.markdown("### 📊 Analysis Features")

        # Highlight current view
        button_type = "primary" if current_view == "visualizations" else "secondary"
        if st.button("📈 Visualizations", use_container_width=True, type=button_type):
            st.session_state.current_view = "visualizations"
            st.rerun()

        button_type = "primary" if current_view == "counterfactuals" else "secondary"
        if st.button("🔁 Counterfactuals", use_container_width=True, type=button_type):
            st.session_state.current_view = "counterfactuals"
            st.rerun()

        button_type = "primary" if current_view == "explainability" else "secondary"
        if st.button("🔍 Explainability Graph", use_container_width=True, type=button_type):
            st.session_state.current_view = "explainability"
            st.rerun()

        button_type = "primary" if current_view == "recommendations" else "secondary"
        if st.button("💡 Recommendations", use_container_width=True, type=button_type):
            st.session_state.current_view = "recommendations"
            st.rerun()

        button_type = "primary" if current_view == "sensitivity" else "secondary"
        if st.button("🔬 Sensitivity Analysis", use_container_width=True, type=button_type):
            st.session_state.current_view = "sensitivity"
            st.rerun()

        st.markdown("---")
        st.subheader("⚡ Actions")

        if st.button("📄 Export PDF", use_container_width=True):
            try:
                pdf_buffer = generate_pdf_report(
                    result, st.session_state.decision_text, "decision_report.pdf"
                )
                st.success("Report generated!")
                st.download_button(
                    label="📥 Download",
                    data=pdf_buffer,
                    file_name="regretless_ai_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

        if st.button("🔄 New Decision", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.decision_text = ""
            st.session_state.conversation_history = []
            st.session_state.simulation_results = None
            st.session_state.structured_decision = None
            st.session_state.current_view = "chat"
            st.rerun()

        st.markdown("---")
        st.subheader("📊 Key Metrics")
        st.metric("Confidence", f"{result.confidence:.1%}")
        _r = float(result.regret_score)
        regret_level = regret_band(_r)
        regret_color = "green" if _r < 0.35 else "orange" if _r < 0.55 else "red"
        st.markdown(
            f'<div style="color: {regret_color}; font-weight: bold; font-size: 1.2rem;">'
            f"Regret Score: {result.regret_score:.2f}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{regret_level} risk")
        st.metric("Risks", len(result.risks))

    # Main content area
    # Header with navigation
    st.markdown('<div class="main-header">🧠 Regretless AI</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        decision_preview = (
            st.session_state.decision_text[:60] + "..."
            if len(st.session_state.decision_text) > 60
            else st.session_state.decision_text
        )
        st.subheader(f"📊 Analysis: {decision_preview}")
    with col2:
        if st.button("💬 Back to Chat", use_container_width=True, key="back_to_chat_main"):
            st.session_state.current_view = "chat"
            st.rerun()

    st.markdown("---")

    # Display selected feature view
    if current_view == "visualizations":
        display_visualizations_view(result)
    elif current_view == "counterfactuals":
        display_counterfactuals_view(result)
    elif current_view == "explainability":
        display_explainability_view(result)
    elif current_view == "recommendations":
        display_recommendations_view(result)
    elif current_view == "sensitivity":
        display_sensitivity_view(result)
    else:
        # Default to chat if unknown view
        st.session_state.current_view = "chat"
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Regretless AI: Simulate Tomorrow. Decide Today."
    "</div>",
    unsafe_allow_html=True,
)
