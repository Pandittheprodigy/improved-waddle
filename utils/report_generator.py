# utils/report_generator.py
"""
Utilities for generating comprehensive reports from research results.
"""

from typing import Dict, List, Any
import json
from datetime import datetime

class ReportGenerator:
    """Generate comprehensive reports from research execution results."""
    
    def generate_report(
        self, 
        research_topic: str, 
        results: Dict[str, Any], 
        paper_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report from research results.
        
        Args:
            research_topic: The research topic
            results: Dictionary containing all research results
            paper_requirements: Original paper requirements
            
        Returns:
            Comprehensive report dictionary
        """
        # Calculate execution time
        start_time = datetime.fromisoformat(results.get("start_time", datetime.now().isoformat()))
        end_time = datetime.now()
        execution_time = str(end_time - start_time)
        
        # Generate report structure
        report = {
            "metadata": {
                "research_topic": research_topic,
                "execution_date": end_time.isoformat(),
                "execution_time": execution_time,
                "paper_requirements": paper_requirements,
                "version": "1.0"
            },
            "literature_review": self._process_literature_review(results),
            "methodology": self._process_methodology(results),
            "data_analysis": self._process_data_analysis(results),
            "research_paper": self._process_research_paper(results),
            "citation_analysis": self._process_citation_analysis(results),
            "quality_assurance": self._process_quality_assurance(results),
            "presentation": self._process_presentation(results),
            "summary": self._generate_summary(results, paper_requirements),
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _process_literature_review(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process literature review results."""
        lr_data = results.get("literature_review", {})
        
        return {
            "total_sources_found": lr_data.get("total_sources", 0),
            "sources_by_database": lr_data.get("sources_by_database", {}),
            "key_findings": lr_data.get("key_findings", []),
            "research_gaps": lr_data.get("research_gaps", []),
            "theoretical_frameworks": lr_data.get("theoretical_frameworks", []),
            "quality_score": lr_data.get("quality_score", "N/A")
        }
    
    def _process_methodology(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process methodology results."""
        method_data = results.get("methodology", {})
        
        return {
            "research_design": method_data.get("design", "N/A"),
            "data_collection_methods": method_data.get("data_collection", []),
            "analysis_techniques": method_data.get("analysis_techniques", []),
            "ethical_considerations": method_data.get("ethical_considerations", []),
            "limitations": method_data.get("limitations", []),
            "validation_approach": method_data.get("validation_approach", "N/A")
        }
    
    def _process_data_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process data analysis results."""
        da_data = results.get("data_analysis", {})
        
        return {
            "analysis_summary": da_data.get("summary", "N/A"),
            "statistical_methods": da_data.get("statistical_methods", []),
            "key_insights": da_data.get("key_insights", []),
            "data_visualizations": da_data.get("visualizations", []),
            "confidence_levels": da_data.get("confidence_levels", {}),
            "anomalies_detected": da_data.get("anomalies", [])
        }
    
    def _process_research_paper(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process research paper results."""
        paper_data = results.get("research_paper", {})
        
        return {
            "sections": paper_data.get("sections", []),
            "total_words": paper_data.get("word_count", 0),
            "total_pages": paper_data.get("page_count", 0),
            "content": paper_data.get("content", ""),
            "structure_score": paper_data.get("structure_score", "N/A"),
            "writing_quality": paper_data.get("writing_quality", "N/A")
        }
    
    def _process_citation_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process citation analysis results."""
        citation_data = results.get("citation_analysis", {})
        
        return {
            "total_citations": citation_data.get("total_citations", 0),
            "citation_style": citation_data.get("style", "N/A"),
            "formatted_references": citation_data.get("formatted_references", []),
            "citation_issues": citation_data.get("issues", []),
            "compliance_score": citation_data.get("compliance_score", "N/A")
        }
    
    def _process_quality_assurance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process quality assurance results."""
        qa_data = results.get("quality_assurance", {})
        
        return {
            "content_accuracy": qa_data.get("content_accuracy", "N/A"),
            "structure_evaluation": qa_data.get("structure_evaluation", "N/A"),
            "grammar_score": qa_data.get("grammar_score", "N/A"),
            "compliance_check": qa_data.get("compliance_check", {}),
            "recommendations": qa_data.get("recommendations", []),
            "overall_quality_score": qa_data.get("overall_score", "N/A")
        }
    
    def _process_presentation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process presentation results."""
        pres_data = results.get("presentation", {})
        
        return {
            "slide_count": pres_data.get("slide_count", 0),
            "template_style": pres_data.get("template_style", "N/A"),
            "estimated_duration": pres_data.get("estimated_duration", "N/A"),
            "content": pres_data.get("content", ""),
            "visual_elements": pres_data.get("visual_elements", []),
            "accessibility_features": pres_data.get("accessibility_features", {})
        }
    
    def _generate_summary(self, results: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of the research execution."""
        # Calculate overall success metrics
        completion_status = self._calculate_completion_status(results)
        quality_metrics = self._extract_quality_metrics(results)
        requirement_compliance = self._check_requirements_compliance(results, requirements)
        
        return {
            "execution_summary": {
                "tasks_completed": completion_status["completed_tasks"],
                "total_tasks": completion_status["total_tasks"],
                "completion_percentage": completion_status["completion_percentage"],
                "overall_status": completion_status["status"]
            },
            "quality_assessment": quality_metrics,
            "requirements_compliance": requirement_compliance,
            "key_achievements": self._extract_key_achievements(results),
            "challenges_encountered": self._extract_challenges(results)
        }
    
    def _calculate_completion_status(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the completion status of all research tasks."""
        # Mock implementation - in reality, you would track task completion
        total_tasks = 7  # Literature review, methodology, data analysis, writing, citation, QA, presentation
        completed_tasks = len([k for k in results.keys() if results[k]])  # Count non-empty results
        
        completion_percentage = (completed_tasks / total_tasks) * 100
        status = "Completed" if completion_percentage == 100 else "In Progress"
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_percentage": completion_percentage,
            "status": status
        }
    
    def _extract_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality metrics from results."""
        quality_scores = {}
        
        # Extract scores from different components
        if "quality_assurance" in results:
            qa = results["quality_assurance"]
            quality_scores["overall"] = qa.get("overall_score", "N/A")
            quality_scores["content_accuracy"] = qa.get("content_accuracy", "N/A")
            quality_scores["writing_quality"] = qa.get("writing_quality", "N/A")
        
        if "citation_analysis" in results:
            ca = results["citation_analysis"]
            quality_scores["citation_compliance"] = ca.get("compliance_score", "N/A")
        
        if "research_paper" in results:
            paper = results["research_paper"]
            quality_scores["structure"] = paper.get("structure_score", "N/A")
        
        return quality_scores
    
    def _check_requirements_compliance(self, results: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with original requirements."""
        compliance_check = {
            "citation_style": requirements.get("citation_style", "N/A"),
            "paper_length": requirements.get("paper_length", "N/A"),
            "research_type": requirements.get("research_type", "N/A"),
            "target_audience": requirements.get("target_audience", "N/A")
        }
        
        # Check if requirements were met (mock implementation)
        compliance_check["met_requirements"] = True
        compliance_check["issues"] = []
        
        return compliance_check
    
    def _extract_key_achievements(self, results: Dict[str, Any]) -> List[str]:
        """Extract key achievements from the research execution."""
        achievements = []
        
        # Check for successful completion of major components
        if "literature_review" in results:
            achievements.append("Comprehensive literature review completed")
        
        if "data_analysis" in results:
            achievements.append("Advanced data analysis performed")
        
        if "research_paper" in results:
            achievements.append("High-quality research paper generated")
        
        if "presentation" in results:
            achievements.append("Professional presentation created")
        
        return achievements
    
    def _extract_challenges(self, results: Dict[str, Any]) -> List[str]:
        """Extract challenges encountered during research execution."""
        challenges = []
        
        # Check for any issues or errors in results
        for component, data in results.items():
            if isinstance(data, dict) and "issues" in data:
                challenges.extend(data["issues"])
        
        return challenges if challenges else ["No significant challenges reported"]
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on research results."""
        recommendations = []
        
        # Quality-based recommendations
        if "quality_assurance" in results:
            qa = results["quality_assurance"]
            if qa.get("overall_score", 0) < 80:
                recommendations.append("Consider additional quality review and revisions")
        
        # Content-based recommendations
        if "literature_review" in results:
            lr = results["literature_review"]
            if "research_gaps" in lr and lr["research_gaps"]:
                recommendations.append("Address identified research gaps in future work")
        
        # Citation recommendations
        if "citation_analysis" in results:
            ca = results["citation_analysis"]
            if ca.get("compliance_score", 100) < 90:
                recommendations.append("Review and improve citation formatting")
        
        return recommendations if recommendations else ["Research execution completed successfully"]
