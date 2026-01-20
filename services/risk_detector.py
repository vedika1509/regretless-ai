"""Service to detect hidden risks and anomalies."""
import numpy as np
import warnings
from typing import List, Dict
from models.scenario import RiskFlag

# Suppress numpy warnings for invalid correlations when variance is zero
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


class RiskDetector:
    """Detect risks in simulation results."""
    
    def detect_risks(self, results: List[Dict[str, float]]) -> List[RiskFlag]:
        """
        Detect risks in simulation results.
        
        Args:
            results: List of simulation results
        
        Returns:
            List of detected risk flags
        """
        risks = []
        
        if not results or len(results) < 10:
            return risks
        
        # Extract metrics
        overall_scores = [r["overall_score"] for r in results]
        risk_scores = [r["risk_score"] for r in results]
        satisfaction_scores = [r["satisfaction_score"] for r in results]
        financial_scores = [r["financial_score"] for r in results]
        
        # 1. High variance (unstable outcome)
        variance = np.var(overall_scores)
        if variance > 0.05:  # Threshold for high variance
            risks.append(RiskFlag(
                risk_type="high_variance",
                severity="medium",
                description="High variance in outcomes suggests unstable results. The decision may have unpredictable consequences."
            ))
        
        # 2. Long-tail downside risk
        mean_score = np.mean(overall_scores)
        percentile_10 = np.percentile(overall_scores, 10)
        downside_gap = mean_score - percentile_10
        
        if downside_gap > 0.3:  # Large gap between mean and worst outcomes
            risks.append(RiskFlag(
                risk_type="long_tail_downside",
                severity="high",
                description=f"Significant downside risk detected. Worst-case scenarios ({percentile_10:.2f}) are substantially worse than average ({mean_score:.2f})."
            ))
        
        # 3. High average risk score
        avg_risk = np.mean(risk_scores)
        if avg_risk > 0.7:
            risks.append(RiskFlag(
                risk_type="high_risk_score",
                severity="high",
                description="High overall risk level detected across scenarios. Consider additional risk mitigation strategies."
            ))
        
        # 4. Negative correlation between satisfaction and financial
        if len(satisfaction_scores) > 10:
            # Check for constant values to avoid division by zero
            sat_std = np.std(satisfaction_scores)
            fin_std = np.std(financial_scores)
            if sat_std > 1e-10 and fin_std > 1e-10:  # Both have non-zero variance
                correlation = np.corrcoef(satisfaction_scores, financial_scores)[0, 1]
                if not np.isnan(correlation) and correlation < -0.3:  # Strong negative correlation
                    risks.append(RiskFlag(
                        risk_type="conflicting_factors",
                        severity="medium",
                        description="Conflicting factors detected: improvements in satisfaction may come at the cost of financial outcomes, or vice versa."
                    ))
        
        # 5. Extreme outliers
        q25 = np.percentile(overall_scores, 25)
        q75 = np.percentile(overall_scores, 75)
        iqr = q75 - q25
        
        outliers = [s for s in overall_scores if s < (q25 - 1.5 * iqr) or s > (q75 + 1.5 * iqr)]
        if len(outliers) > len(results) * 0.05:  # More than 5% outliers
            risks.append(RiskFlag(
                risk_type="extreme_outliers",
                severity="low",
                description="Unusual extreme outcomes detected in some simulations. While rare, these scenarios should be considered."
            ))
        
        # 6. Low satisfaction despite good financials
        high_financial = [r for r in results if r["financial_score"] > 0.7]
        if high_financial:
            avg_satisfaction_high_fin = np.mean([r["satisfaction_score"] for r in high_financial])
            if avg_satisfaction_high_fin < 0.4:
                risks.append(RiskFlag(
                    risk_type="financial_satisfaction_mismatch",
                    severity="medium",
                    description="Despite positive financial outcomes, satisfaction scores remain low. This suggests non-financial factors are important."
                ))
        
        return risks
