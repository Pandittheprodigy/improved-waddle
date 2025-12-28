# tools/research_tools.py (Updated with proper imports)
"""
Custom tools for research tasks in the Harvard Research Paper Publication Crew.

This module contains specialized tools for academic research, literature review,
citation checking, plagiarism detection, and other research-related tasks.
"""

from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
import requests
import json
import os
from datetime import datetime

class AcademicSearchTool(BaseTool):
    """Tool for conducting academic literature searches."""
    
    name: str = "Academic Search Tool"
    description: str = "Search for academic papers, articles, and research materials across multiple databases."
    
    def _run(self, query: str, sources: List[str] = None, max_results: int = 20) -> Dict[str, Any]:
        """
        Perform academic search across specified sources.
        
        Args:
            query: Search query string
            sources: List of academic databases to search (e.g., ["Google Scholar", "PubMed", "IEEE Xplore"])
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        if sources is None:
            sources = ["Google Scholar", "Semantic Scholar", "arXiv"]
        
        results = {}
        
        for source in sources:
            try:
                # Simulate search for each source
                search_results = self._search_source(query, source, max_results)
                results[source] = search_results
            except Exception as e:
                results[source] = {"error": str(e)}
        
        return {
            "query": query,
            "sources": sources,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _search_source(self, query: str, source: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform search on a specific academic source."""
        # This is a placeholder implementation
        # In a real implementation, you would integrate with actual APIs
        
        if source == "Google Scholar":
            # Mock results for demonstration
            return [
                {
                    "title": f"Research Paper on {query} - Part 1",
                    "authors": ["John Doe", "Jane Smith"],
                    "year": 2023,
                    "journal": "Academic Journal of Research",
                    "abstract": "This is a comprehensive study on the topic of " + query,
                    "url": f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
                }
            ]
        
        elif source == "Semantic Scholar":
            return [
                {
                    "title": f"Semantic Analysis of {query}",
                    "authors": ["Alice Johnson", "Bob Wilson"],
                    "year": 2022,
                    "journal": "Semantic Research Quarterly",
                    "abstract": "Advanced semantic analysis techniques applied to " + query,
                    "url": f"https://www.semanticscholar.org/search?q={query.replace(' ', '+')}"
                }
            ]
        
        elif source == "arXiv":
            return [
                {
                    "title": f"Preprint: Novel Approaches to {query}",
                    "authors": ["Research Team"],
                    "year": 2024,
                    "journal": "arXiv Preprint",
                    "abstract": "Cutting-edge research on " + query,
                    "url": f"https://arxiv.org/search/?query={query.replace(' ', '+')}"
                }
            ]
        
        return []

class CitationCheckerTool(BaseTool):
    """Tool for validating and formatting citations."""
    
    name: str = "Citation Checker Tool"
    description: str = "Validate citations and format them according to specified citation styles (APA, MLA, Chicago, etc.)."
    
    def _run(self, citations: List[Dict[str, Any]], style: str = "APA") -> Dict[str, Any]:
        """
        Validate and format citations according to the specified style.
        
        Args:
            citations: List of citation dictionaries with raw citation data
            style: Citation style (APA, MLA, Chicago, Harvard, IEEE)
            
        Returns:
            Dictionary containing formatted citations and validation results
        """
        formatted_citations = []
        validation_results = []
        
        for i, citation in enumerate(citations):
            try:
                formatted = self._format_citation(citation, style)
                validated = self._validate_citation(citation)
                
                formatted_citations.append(formatted)
                validation_results.append({
                    "citation_id": i,
                    "valid": validated["valid"],
                    "issues": validated["issues"],
                    "suggestions": validated["suggestions"]
                })
            except Exception as e:
                validation_results.append({
                    "citation_id": i,
                    "valid": False,
                    "issues": [str(e)],
                    "suggestions": []
                })
        
        return {
            "style": style,
            "original_count": len(citations),
            "formatted_citations": formatted_citations,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_citation(self, citation: Dict[str, Any], style: str) -> str:
        """Format a single citation according to the specified style."""
        authors = citation.get("authors", [])
        title = citation.get("title", "")
        year = citation.get("year", "")
        journal = citation.get("journal", "")
        volume = citation.get("volume", "")
        issue = citation.get("issue", "")
        pages = citation.get("pages", "")
        doi = citation.get("doi", "")
        
        if style.upper() == "APA":
            # Format: Author, A. A. (Year). Title of article. Title of Journal, volume(issue), pages.
            author_str = self._format_authors_apa(authors)
            return f"{author_str} ({year}). {title}. {journal}, {volume}({issue}), {pages}."
        
        elif style.upper() == "MLA":
            # Format: Author(s). "Title of Article." Title of Journal, vol. number, no. number, year, pages.
            author_str = self._format_authors_mla(authors)
            return f'{author_str} "{title}." {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}.'
        
        elif style.upper() == "CHICAGO":
            # Format: Author(s). Year. "Title of Article." Title of Journal volume, no. issue (Month): pages.
            author_str = self._format_authors_chicago(authors)
            return f'{author_str} {year}. "{title}." {journal} {volume}, no. {issue} ({self._get_month()}): {pages}.'
        
        else:
            # Default format
            return f"{', '.join(authors)}. {title}. {journal}, {year}."
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors list for APA style."""
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        else:
            return f"{authors[0]} et al."
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors list for MLA style."""
        if len(authors) <= 3:
            return " and ".join(authors)
        else:
            return f"{authors[0]} et al."
    
    def _format_authors_chicago(self, authors: List[str]) -> str:
        """Format authors list for Chicago style."""
        if len(authors) <= 3:
            return ", ".join(authors)
        else:
            return f"{authors[0]} et al."
    
    def _validate_citation(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a citation for completeness and accuracy."""
        issues = []
        suggestions = []
        
        required_fields = ["authors", "title", "year", "journal"]
        for field in required_fields:
            if not citation.get(field):
                issues.append(f"Missing required field: {field}")
                suggestions.append(f"Please provide the {field}.")
        
        # Check year format
        year = citation.get("year", "")
        if year and not year.isdigit():
            issues.append("Invalid year format")
            suggestions.append("Please provide the year as a 4-digit number.")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _get_month(self) -> str:
        """Get current month name for citation formatting."""
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return months[datetime.now().month - 1]

class PlagiarismCheckerTool(BaseTool):
    """Tool for checking content for plagiarism."""
    
    name: str = "Plagiarism Checker Tool"
    description: str = "Check written content for potential plagiarism and generate similarity reports."
    
    def _run(self, content: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        Check content for plagiarism against specified sources.
        
        Args:
            content: The content to check for plagiarism
            sources: List of sources to check against
            
        Returns:
            Dictionary containing plagiarism analysis results
        """
        if sources is None:
            sources = ["Web", "Academic Databases", "Published Works"]
        
        # Simulate plagiarism check
        # In a real implementation, you would integrate with plagiarism detection APIs
        
        similarity_scores = {}
        for source in sources:
            # Generate mock similarity scores
            import random
            score = random.randint(5, 25)  # Random score between 5% and 25%
            similarity_scores[source] = score
        
        total_similarity = sum(similarity_scores.values()) / len(similarity_scores)
        
        # Identify potential issues
        issues = []
        if total_similarity > 20:
            issues.append("High similarity detected. Review content for proper citations.")
        elif total_similarity > 10:
            issues.append("Moderate similarity detected. Consider paraphrasing.")
        
        return {
            "content_length": len(content),
            "sources_checked": sources,
            "similarity_scores": similarity_scores,
            "total_similarity": total_similarity,
            "issues": issues,
            "recommendations": self._get_recommendations(total_similarity),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_recommendations(self, similarity: float) -> List[str]:
        """Get recommendations based on similarity score."""
        if similarity > 25:
            return [
                "High similarity detected. Review all sections for proper citation.",
                "Consider significant paraphrasing of similar content.",
                "Ensure all sources are properly attributed."
            ]
        elif similarity > 15:
            return [
                "Moderate similarity detected. Review highlighted sections.",
                "Add proper citations where needed.",
                "Consider minor paraphrasing for improvement."
            ]
        else:
            return [
                "Similarity level is acceptable.",
                "Ensure all sources are properly cited.",
                "Continue with current writing approach."
            ]

class LiteratureReviewTool(BaseTool):
    """Tool for conducting systematic literature reviews."""
    
    name: str = "Literature Review Tool"
    description: str = "Conduct systematic literature reviews and synthesize research findings."
    
    def _run(
        self, 
        research_question: str, 
        inclusion_criteria: List[str], 
        exclusion_criteria: List[str], 
        databases: List[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct a systematic literature review.
        
        Args:
            research_question: The research question to guide the review
            inclusion_criteria: List of criteria for including studies
            exclusion_criteria: List of criteria for excluding studies
            databases: List of databases to search
            
        Returns:
            Dictionary containing literature review results
        """
        if databases is None:
            databases = ["PubMed", "IEEE Xplore", "Google Scholar", "Scopus"]
        
        # Simulate literature review process
        # In a real implementation, you would integrate with actual database APIs
        
        search_results = []
        for db in databases:
            # Mock search results
            results = self._search_database(db, research_question)
            search_results.append({
                "database": db,
                "results_count": len(results),
                "studies": results
            })
        
        # Apply inclusion/exclusion criteria
        included_studies = []
        excluded_studies = []
        
        for db_results in search_results:
            for study in db_results["studies"]:
                if self._meets_criteria(study, inclusion_criteria, exclusion_criteria):
                    included_studies.append(study)
                else:
                    excluded_studies.append({
                        "study": study,
                        "reason": self._get_exclusion_reason(study, exclusion_criteria)
                    })
        
        # Synthesize findings
        synthesis = self._synthesize_findings(included_studies)
        
        return {
            "research_question": research_question,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "databases_searched": databases,
            "search_results": search_results,
            "included_studies": included_studies,
            "excluded_studies": excluded_studies,
            "findings_synthesis": synthesis,
            "research_gaps": self._identify_research_gaps(included_studies),
            "timestamp": datetime.now().isoformat()
        }
    
    def _search_database(self, database: str, query: str) -> List[Dict[str, Any]]:
        """Search a specific database for studies related to the query."""
        # Mock implementation
        return [
            {
                "title": f"Study on {query} in {database}",
                "authors": ["Researcher 1", "Researcher 2"],
                "year": 2023,
                "abstract": f"This study investigates {query} using methods specific to {database}.",
                "methodology": "Quantitative analysis",
                "sample_size": 100,
                "key_findings": [f"Finding 1 related to {query}", f"Finding 2 related to {query}"]
            }
        ]
    
    def _meets_criteria(self, study: Dict[str, Any], inclusion: List[str], exclusion: List[str]) -> bool:
        """Check if a study meets the inclusion and exclusion criteria."""
        # Mock implementation - in reality, this would analyze the study content
        return True
    
    def _get_exclusion_reason(self, study: Dict[str, Any], exclusion_criteria: List[str]) -> str:
        """Get the reason for excluding a study."""
        return "Did not meet inclusion criteria"
    
    def _synthesize_findings(self, studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize findings from included studies."""
        if not studies:
            return {"summary": "No studies included for synthesis", "themes": [], "conclusions": []}
        
        # Extract themes and patterns
        themes = []
        conclusions = []
        
        for study in studies:
            # Mock theme extraction
            themes.extend([f"Theme from {study['title'][:50]}..."])
            conclusions.extend(study.get("key_findings", []))
        
        return {
            "number_of_studies": len(studies),
            "themes": list(set(themes)),  # Remove duplicates
            "common_conclusions": conclusions[:5],  # Limit to first 5
            "methodological_patterns": ["Common methods identified"],
            "quality_assessment": "All studies met quality criteria"
        }
    
    def _identify_research_gaps(self, studies: List[Dict[str, Any]]) -> List[str]:
        """Identify potential research gaps based on the literature review."""
        if not studies:
            return ["No studies found - significant research gap identified"]
        
        # Mock gap identification
        return [
            "Limited longitudinal studies in this area",
            "Need for more diverse sample populations",
            "Gap in qualitative research approaches",
            "Insufficient attention to emerging technologies"
        ]
