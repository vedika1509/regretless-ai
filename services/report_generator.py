"""Generate PDF reports."""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from models.scenario import AnalysisResult
from utils.humanize import band_label, compute_readiness, regret_band, to_10


def generate_pdf_report(result: AnalysisResult, decision_text: str, filename: str = None) -> BytesIO:
    """Generate a PDF report of the analysis."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#FF4B4B'),
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#262730'),
        spaceAfter=12
    )
    
    # Title
    title = Paragraph("Regretless AI Decision Analysis Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Decision
    decision_para = Paragraph(f"<b>Decision:</b> {decision_text}", styles['Normal'])
    story.append(decision_para)
    story.append(Spacer(1, 0.3*inch))
    
    # Confidence
    conf_para = Paragraph(
        f"<b>Confidence:</b> {result.confidence:.1%}",
        styles['Normal']
    )
    story.append(conf_para)
    story.append(Spacer(1, 0.2*inch))

    # Decision readiness + human metrics (most likely)
    try:
        verdict = compute_readiness(
            regret01=float(result.regret_score),
            confidence01=float(result.confidence),
            risks=result.risks,
            context={},  # context already baked into decision_text for this export
        )
        story.append(
            Paragraph(
                f"<b>Decision Readiness:</b> {verdict.label}<br/><i>{verdict.rationale}</i>",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        most = result.scenarios.get("most_likely")
        if most:
            likely_label, _ = band_label(float(most.outcomes.overall_score))
            job_sec = float((most.metrics or {}).get("job_security", 0.5))
            fin = float((most.metrics or {}).get("financial_satisfaction", most.outcomes.financial_score))
            stress = float((most.metrics or {}).get("stress", 0.5))
            mental = 1.0 - stress

            story.append(Paragraph(f"<b>Likely Outcome:</b> {likely_label}", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

            metrics_data = [
                ["Human Metric", "Value"],
                ["Career Stability", f"{to_10(job_sec)}/10"],
                ["Financial Safety", f"{to_10(fin)}/10"],
                ["Mental Wellbeing", f"{to_10(mental)}/10"],
                ["Long-term Regret Risk", regret_band(float(result.regret_score))],
            ]
            metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
            metrics_table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            story.append(metrics_table)
            story.append(Spacer(1, 0.25 * inch))
    except Exception:
        # Keep PDF export resilient even if interpretability helpers fail
        pass
    
    # Simulation info
    sim_para = Paragraph(
        f"<b>Simulations Run:</b> {result.simulation_count:,}",
        styles['Normal']
    )
    story.append(sim_para)
    story.append(Spacer(1, 0.3*inch))
    
    # Scenarios
    story.append(Paragraph("<b>Scenarios:</b>", heading_style))
    
    scenario_names = {
        "best_case": "Best Case",
        "worst_case": "Worst Case",
        "most_likely": "Most Likely"
    }
    
    for scenario_type, scenario in result.scenarios.items():
        scenario_name = scenario_names.get(scenario_type, scenario_type.replace('_', ' ').title())
        
        scenario_header = Paragraph(
            f"<b>{scenario_name} (Probability: {scenario.probability:.1%})</b>",
            styles['Heading3']
        )
        story.append(scenario_header)
        
        # Metrics table
        metrics_data = [
            ['Metric', 'Score'],
            ['Financial Score', f"{scenario.outcomes.financial_score:.2f}"],
            ['Satisfaction Score', f"{scenario.outcomes.satisfaction_score:.2f}"],
            ['Risk Score', f"{scenario.outcomes.risk_score:.2f}"],
            ['Overall Score', f"{scenario.outcomes.overall_score:.2f}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        story.append(metrics_table)
        story.append(Spacer(1, 0.1*inch))
        
        # Explanation
        explanation_para = Paragraph(
            f"<b>Explanation:</b> {scenario.explanation}",
            styles['Normal']
        )
        story.append(explanation_para)
        story.append(Spacer(1, 0.2*inch))
    
    # Risks
    if result.risks:
        story.append(Paragraph("<b>Detected Risks:</b>", heading_style))
        for risk in result.risks:
            risk_para = Paragraph(
                f"<b>{risk.risk_type.replace('_', ' ').title()} ({risk.severity.upper()}):</b> {risk.description}",
                styles['Normal']
            )
            story.append(risk_para)
            story.append(Spacer(1, 0.1*inch))
    else:
        story.append(Paragraph("<b>Risks:</b> No significant risks detected.", heading_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer = Paragraph(
        "<i>Generated by Regretless AI: Simulate Tomorrow. Decide Today.</i>",
        styles['Italic']
    )
    story.append(footer)
    
    doc.build(story)
    buffer.seek(0)
    
    # If filename provided, also save to file
    if filename:
        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())
        buffer.seek(0)
    
    return buffer
