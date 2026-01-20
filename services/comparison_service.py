"""Service for comparing multiple decisions."""
from typing import List, Dict
from models.scenario import AnalysisResult


class DecisionComparator:
    """Compare multiple decision analyses."""
    
    def compare_decisions(self, decisions: List[Dict]) -> Dict:
        """
        Compare multiple decisions.
        
        Args:
            decisions: List of {"name": str, "result": AnalysisResult}
        
        Returns:
            Comparison analysis
        """
        comparison = {
            "best_overall": None,
            "lowest_risk": None,
            "highest_confidence": None,
            "tradeoffs": [],
            "scores": {}
        }
        
        if not decisions or len(decisions) < 2:
            return comparison
        
        # Find best overall score
        best_score = -1
        best_name = None
        for decision in decisions:
            avg_score = sum(
                s.outcomes.overall_score for s in decision["result"].scenarios.values()
            ) / 3
            comparison["scores"][decision["name"]] = {
                "overall_avg": avg_score,
                "confidence": decision["result"].confidence
            }
            if avg_score > best_score:
                best_score = avg_score
                best_name = decision["name"]
        comparison["best_overall"] = best_name
        
        # Find lowest risk
        lowest_risk = 1.0
        lowest_risk_name = None
        for decision in decisions:
            avg_risk = sum(
                s.outcomes.risk_score for s in decision["result"].scenarios.values()
            ) / 3
            if avg_risk < lowest_risk:
                lowest_risk = avg_risk
                lowest_risk_name = decision["name"]
        comparison["lowest_risk"] = lowest_risk_name
        
        # Find highest confidence
        highest_conf = -1
        highest_conf_name = None
        for decision in decisions:
            if decision["result"].confidence > highest_conf:
                highest_conf = decision["result"].confidence
                highest_conf_name = decision["name"]
        comparison["highest_confidence"] = highest_conf_name
        
        # Identify tradeoffs
        if len(decisions) >= 2:
            # Check if one has better score but higher risk
            for i, decision1 in enumerate(decisions):
                for decision2 in decisions[i+1:]:
                    avg_score1 = sum(
                        s.outcomes.overall_score for s in decision1["result"].scenarios.values()
                    ) / 3
                    avg_score2 = sum(
                        s.outcomes.overall_score for s in decision2["result"].scenarios.values()
                    ) / 3
                    avg_risk1 = sum(
                        s.outcomes.risk_score for s in decision1["result"].scenarios.values()
                    ) / 3
                    avg_risk2 = sum(
                        s.outcomes.risk_score for s in decision2["result"].scenarios.values()
                    ) / 3
                    
                    if avg_score1 > avg_score2 and avg_risk1 > avg_risk2:
                        comparison["tradeoffs"].append(
                            f"{decision1['name']} has better outcomes but higher risk than {decision2['name']}"
                        )
                    elif avg_score2 > avg_score1 and avg_risk2 > avg_risk1:
                        comparison["tradeoffs"].append(
                            f"{decision2['name']} has better outcomes but higher risk than {decision1['name']}"
                        )
        
        return comparison
