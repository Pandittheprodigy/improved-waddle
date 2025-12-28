# agents/research_agents.py
"""
Agent classes for the Harvard Research Paper Publication Crew.

This module contains specialized agent classes for different aspects of
research paper creation and presentation generation.
"""

from typing import Dict, Any, List
from crewai import Agent
from langchain.tools import BaseTool
from tools.research_tools import (
    AcademicSearchTool, 
    CitationCheckerTool, 
    PlagiarismCheckerTool,
    LiteratureReviewTool
)
from tools.presentation_tools import (
    PowerPointPresentationTool,
    VisualDesignTool,
    DataVisualizationTool
)

class BaseAgent:
    """Base class for all specialized agents."""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str, api_keys: Dict[str, str]):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.api_keys = api_keys
    
    def create_agent(self) -> Agent:
        """Create and return a CrewAI Agent instance."""
        raise NotImplementedError("Subclasses must implement create_agent method")

class ResearchCoordinator(BaseAgent):
    """Agent responsible for coordinating the entire research process."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[AcademicSearchTool()],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_keys["gemini_api_key"],
            temperature=0.7
        )

class LiteratureReviewer(BaseAgent):
    """Agent specialized in conducting literature reviews."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[
                AcademicSearchTool(),
                LiteratureReviewTool(),
                SerperDevTool()
            ],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_keys["openrouter_api_key"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.5
        )

class DataAnalyst(BaseAgent):
    """Agent specialized in data analysis and statistical methods."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[DataVisualizationTool()],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_keys["gemini_api_key"],
            temperature=0.3
        )

class MethodologyExpert(BaseAgent):
    """Agent specialized in research methodology design."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[CodeDocsSearchTool()],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_keys["openrouter_api_key"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.5
        )

class WritingSpecialist(BaseAgent):
    """Agent specialized in academic writing."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[
                FileWriteTool(),
                FileReadTool(),
                DirectorySearchTool()
            ],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_keys["gemini_api_key"],
            temperature=0.7
        )

class CitationExpert(BaseAgent):
    """Agent specialized in citation management and reference formatting."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[CitationCheckerTool()],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_keys["openrouter_api_key"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3
        )

class QualityAssurance(BaseAgent):
    """Agent responsible for quality control and validation."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[PlagiarismCheckerTool()],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_keys["gemini_api_key"],
            temperature=0.2
        )

class PresentationExpert(BaseAgent):
    """Agent specialized in creating professional presentations."""
    
    def create_agent(self) -> Agent:
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            memory=True,
            tools=[
                PowerPointPresentationTool(),
                VisualDesignTool(),
                DataVisualizationTool()
            ],
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """Get the LLM for this agent."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_keys["openrouter_api_key"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )
