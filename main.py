# main.py (Fixed Version)
"""
Harvard Research Paper Publication Crew

A comprehensive AI agent system for research paper creation and presentation generation.
This system uses CrewAI for agent orchestration and Streamlit for the UI.
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import tempfile
import zipfile
import base64
from io import BytesIO
import logging

# Import CrewAI components
from crewai import Agent, Task, Crew, Process

# Import custom tools
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

# Import agent classes
from agents.research_agents import (
    ResearchCoordinator,
    LiteratureReviewer,
    DataAnalyst,
    MethodologyExpert,
    WritingSpecialist,
    CitationExpert,
    QualityAssurance,
    PresentationExpert
)

# Import utilities
from utils.report_generator import ReportGenerator
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# Import file tools from langchain instead
from langchain.tools import FileReadTool, WriteFileTool
from langchain_community.tools import DuckDuckGoSearchResults

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger("harvard_crew", "logs/harvard_crew.log")

class HarvardResearchCrew:
    """
    Main class for the Harvard Research Paper Publication Crew.
    
    This class orchestrates multiple specialized AI agents to create
    high-quality research papers and presentations.
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.api_keys = self._load_api_keys()
        self.crew = None
        self.agents = []
        self.tasks = []
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        return {
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "groq_api_key": os.getenv("GROQ_API_KEY", ""),
            "serper_api_key": os.getenv("SERPER_API_KEY", ""),
        }
    
    def _validate_api_keys(self) -> bool:
        """Validate that all required API keys are present."""
        required_keys = ["gemini_api_key", "openrouter_api_key", "groq_api_key", "serper_api_key"]
        missing_keys = [key for key in required_keys if not self.api_keys[key]]
        
        if missing_keys:
            logger.error(f"Missing API keys: {', '.join(missing_keys)}")
            return False
        return True
    
    def setup_agents(self, research_topic: str, paper_requirements: Dict[str, Any]) -> List[Agent]:
        """
        Initialize all specialized agents for the research crew.
        
        Args:
            research_topic: The main topic of the research paper
            paper_requirements: Requirements and specifications for the paper
            
        Returns:
            List of initialized Agent objects
        """
        try:
            # Initialize specialized agents
            research_coordinator = ResearchCoordinator(
                name="Research Coordinator",
                role="Lead Research Coordinator",
                goal="Orchestrate the entire research process and ensure all components work together seamlessly",
                backstory="Experienced research coordinator with a background in managing complex academic projects",
                api_keys=self.api_keys
            )
            
            literature_reviewer = LiteratureReviewer(
                name="Literature Reviewer", 
                role="Academic Literature Specialist",
                goal="Conduct comprehensive literature reviews and identify relevant research",
                backstory="PhD in academic research with extensive experience in literature analysis",
                api_keys=self.api_keys
            )
            
            data_analyst = DataAnalyst(
                name="Data Analyst",
                role="Data Science and Statistics Expert", 
                goal="Analyze research data and provide statistical insights",
                backstory="Data scientist with a focus on academic research methodologies",
                api_keys=self.api_keys
            )
            
            methodology_expert = MethodologyExpert(
                name="Methodology Expert",
                role="Research Methodology Consultant",
                goal="Design robust research methodologies and validate approaches",
                backstory="Methodology consultant with expertise in various research frameworks",
                api_keys=self.api_keys
            )
            
            writing_specialist = WritingSpecialist(
                name="Writing Specialist",
                role="Academic Writing Expert",
                goal="Write and edit the research paper with proper academic style and structure",
                backstory="Professional academic writer with Harvard-level experience",
                api_keys=self.api_keys
            )
            
            citation_expert = CitationExpert(
                name="Citation Expert",
                role="Citation and Reference Specialist",
                goal="Ensure all citations and references follow proper academic standards",
                backstory="Reference management specialist with experience in various citation styles",
                api_keys=self.api_keys
            )
            
            quality_assurance = QualityAssurance(
                name="Quality Assurance",
                role="Quality Control Specialist",
                goal="Review and validate all aspects of the research paper for quality and accuracy",
                backstory="Quality assurance expert with a keen eye for detail in academic work",
                api_keys=self.api_keys
            )
            
            presentation_expert = PresentationExpert(
                name="Presentation Expert",
                role="Presentation and Visualization Specialist",
                goal="Create compelling PowerPoint presentations based on the research findings",
                backstory="Professional presentation designer with experience in academic conferences",
                api_keys=self.api_keys
            )
            
            self.agents = [
                research_coordinator.create_agent(),
                literature_reviewer.create_agent(),
                data_analyst.create_agent(),
                methodology_expert.create_agent(),
                writing_specialist.create_agent(),
                citation_expert.create_agent(),
                quality_assurance.create_agent(),
                presentation_expert.create_agent()
            ]
            
            logger.info(f"Successfully initialized {len(self.agents)} agents for research on: {research_topic}")
            return self.agents
            
        except Exception as e:
            logger.error(f"Error setting up agents: {str(e)}")
            raise
    
    def setup_tasks(self, research_topic: str, paper_requirements: Dict[str, Any]) -> List[Task]:
        """
        Create and configure all tasks for the research crew.
        
        Args:
            research_topic: The main topic of the research paper
            paper_requirements: Requirements and specifications for the paper
            
        Returns:
            List of Task objects
        """
        try:
            # Define all tasks with proper tool imports
            file_read_tool = FileReadTool()
            file_write_tool = WriteFileTool()
            web_search_tool = DuckDuckGoSearchResults()
            
            task_1 = Task(
                description=(
                    f"Conduct comprehensive literature review on '{research_topic}'. "
                    "Identify key papers, theories, and research gaps. "
                    "Focus on recent publications (last 5 years) and seminal works. "
                    "Provide summary of findings with proper citations."
                ),
                expected_output=(
                    "Detailed literature review report including:\n"
                    "- Summary of key findings\n"
                    "- Identified research gaps\n"
                    "- Relevant theoretical frameworks\n"
                    "- Proper citations in the required format"
                ),
                agent=self.agents[1],  # Literature Reviewer
                tools=[AcademicSearchTool(), LiteratureReviewTool(), web_search_tool],
                async_execution=True
            )
            
            task_2 = Task(
                description=(
                    f"Design research methodology for '{research_topic}' based on the literature review. "
                    "Determine appropriate research methods, data collection techniques, "
                    "and analysis approaches. Ensure methodology aligns with research objectives."
                ),
                expected_output=(
                    "Comprehensive methodology section including:\n"
                    "- Research design justification\n"
                    "- Data collection methods\n"
                    "- Analysis techniques\n"
                    "- Ethical considerations\n"
                    "- Limitations discussion"
                ),
                agent=self.agents[3],  # Methodology Expert
                tools=[web_search_tool],
                async_execution=True
            )
            
            task_3 = Task(
                description=(
                    f"Analyze available data for '{research_topic}' research. "
                    "Apply appropriate statistical methods and data analysis techniques. "
                    "Generate insights and findings based on the data analysis."
                ),
                expected_output=(
                    "Data analysis report with:\n"
                    "- Statistical analysis results\n"
                    "- Data visualizations\n"
                    "- Key findings summary\n"
                    "- Interpretation of results"
                ),
                agent=self.agents[2],  # Data Analyst
                tools=[DataVisualizationTool()],
                async_execution=True
            )
            
            task_4 = Task(
                description=(
                    f"Write the research paper on '{research_topic}' incorporating findings from "
                    "literature review, data analysis, and methodology. Ensure academic writing "
                    "style and proper structure. Follow the specified formatting requirements."
                ),
                expected_output=(
                    "Complete research paper including:\n"
                    "- Abstract\n"
                    "- Introduction\n"
                    "- Literature review\n"
                    "- Methodology\n"
                    "- Results\n"
                    "- Discussion\n"
                    "- Conclusion\n"
                    "- References"
                ),
                agent=self.agents[4],  # Writing Specialist
                tools=[file_write_tool, file_read_tool],
                async_execution=False
            )
            
            task_5 = Task(
                description=(
                    f"Review and format all citations and references for the '{research_topic}' paper. "
                    "Ensure compliance with the specified citation style (APA, MLA, Chicago, etc.). "
                    "Check for any missing or incorrect citations."
                ),
                expected_output=(
                    "Formatted reference list and in-text citations that comply with the "
                    "specified citation style. All sources properly cited and referenced."
                ),
                agent=self.agents[5],  # Citation Expert
                tools=[CitationCheckerTool()],
                async_execution=False
            )
            
            task_6 = Task(
                description=(
                    f"Conduct quality assurance review of the complete '{research_topic}' research paper. "
                    "Check for content accuracy, logical flow, grammar, and adherence to academic standards. "
                    "Ensure all requirements in paper_requirements are met."
                ),
                expected_output=(
                    "Quality assurance report with:\n"
                    "- Content accuracy assessment\n"
                    "- Structure and flow evaluation\n"
                    "- Grammar and style review\n"
                    "- Compliance check against requirements\n"
                    "- Recommendations for improvements"
                ),
                agent=self.agents[6],  # Quality Assurance
                tools=[PlagiarismCheckerTool()],
                async_execution=False
            )
            
            task_7 = Task(
                description=(
                    f"Create a professional PowerPoint presentation based on the '{research_topic}' research paper. "
                    "Design compelling slides that effectively communicate the research findings to an academic audience. "
                    "Include appropriate visualizations, charts, and key points."
                ),
                expected_output=(
                    "Professional PowerPoint presentation with:\n"
                    "- Title slide with research details\n"
                    "- Introduction and background slides\n"
                    "- Methodology overview\n"
                    "- Key findings presentation\n"
                    "- Data visualizations and charts\n"
                    "- Conclusion and implications\n"
                    "- References slide"
                ),
                agent=self.agents[7],  # Presentation Expert
                tools=[PowerPointPresentationTool(), VisualDesignTool()],
                async_execution=False
            )
            
            # Orchestrate tasks with dependencies
            task_1 >> task_2 >> task_3 >> task_4 >> task_5 >> task_6 >> task_7
            
            self.tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]
            logger.info(f"Successfully created {len(self.tasks)} tasks for research on: {research_topic}")
            return self.tasks
            
        except Exception as e:
            logger.error(f"Error setting up tasks: {str(e)}")
            raise
    
    def create_crew(self, research_topic: str, paper_requirements: Dict[str, Any]) -> Crew:
        """
        Create and configure the CrewAI crew with agents and tasks.
        
        Args:
            research_topic: The main topic of the research paper
            paper_requirements: Requirements and specifications for the paper
            
        Returns:
            Crew object ready for execution
        """
        try:
            if not self._validate_api_keys():
                st.error("Missing required API keys. Please configure them in the sidebar.")
                return None
            
            # Setup agents
            agents = self.setup_agents(research_topic, paper_requirements)
            
            # Setup tasks
            tasks = self.setup_tasks(research_topic, paper_requirements)
            
            # Create crew
            self.crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.hierarchical,
                manager_llm=self._get_manager_llm(),
                verbose=True,
                max_rpm=100,
                function_calling_llm=self._get_function_calling_llm()
            )
            
            logger.info("Successfully created Harvard Research Crew")
            return self.crew
            
        except Exception as e:
            logger.error(f"Error creating crew: {str(e)}")
            st.error(f"Error creating research crew: {str(e)}")
            return None
    
    def _get_manager_llm(self):
        """Get the LLM for crew management."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_keys["gemini_api_key"],
            temperature=0.7
        )
    
    def _get_function_calling_llm(self):
        """Get the LLM for function calling."""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_keys["openrouter_api_key"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.5
        )
    
    async def execute_research(self, research_topic: str, paper_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete research process.
        
        Args:
            research_topic: The main topic of the research paper
            paper_requirements: Requirements and specifications for the paper
            
        Returns:
            Dictionary containing results and outputs
        """
        try:
            # Create crew
            crew = self.create_crew(research_topic, paper_requirements)
            
            if crew is None:
                return {"error": "Failed to create research crew"}
            
            # Execute the crew
            logger.info(f"Starting research execution for: {research_topic}")
            start_time = datetime.now()
            
            # Use asyncio.create_task for async execution
            execution_task = asyncio.create_task(crew.kickoff_async())
            
            # Monitor progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates during execution
            for i in range(100):
                if execution_task.done():
                    break
                progress_bar.progress(i + 1)
                status_text.text(f"Research in progress: {i + 1}%")
                await asyncio.sleep(0.1)
            
            # Get results
            results = await execution_task
            
            end_time = datetime.now()
            execution_time = str(end_time - start_time)
            
            # Generate comprehensive report
            report_generator = ReportGenerator()
            final_report = report_generator.generate_report(
                research_topic=research_topic,
                results=results,
                paper_requirements=paper_requirements
            )
            
            # Add execution metadata
            final_report["execution_metadata"] = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "research_topic": research_topic
            }
            
            logger.info("Research execution completed successfully")
            return final_report
            
        except Exception as e:
            logger.error(f"Error executing research: {str(e)}")
            return {"error": str(e), "research_topic": research_topic}

# Streamlit UI Application
def main():
    """Main Streamlit application for the Harvard Research Paper Publication Crew."""
    
    # Set page configuration
    st.set_page_config(
        page_title="Harvard Research Paper Publication Crew",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            border-left: 4px solid #1f77b4;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .agent-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 0.5rem;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #28a745;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">Harvard Research Paper Publication Crew</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🚀 Configuration")
        
        # API Key Inputs
        st.subheader("🔑 API Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
        groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        serper_api_key = st.text_input("Serper API Key", type="password", value=os.getenv("SERPER_API_KEY", ""))
        
        # Save API keys to environment
        if st.button("💾 Save API Keys"):
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
            os.environ["GROQ_API_KEY"] = groq_api_key
            os.environ["SERPER_API_KEY"] = serper_api_key
            st.success("✅ API keys saved successfully!")
        
        # Research Configuration
        st.subheader("📋 Research Configuration")
        citation_style = st.selectbox("Citation Style", ["APA", "MLA", "Chicago", "Harvard", "IEEE"])
        paper_length = st.selectbox("Paper Length", ["Short (5-10 pages)", "Medium (10-20 pages)", "Long (20+ pages)"])
        target_journal = st.text_input("Target Journal/Conference", placeholder="e.g., Nature, Science, IEEE Transactions")
        
        # Advanced Options
        st.subheader("⚙️ Advanced Options")
        enable_plagiarism_check = st.checkbox("Enable Plagiarism Check", value=True)
        enable_data_analysis = st.checkbox("Enable Data Analysis", value=True)
        enable_presentation = st.checkbox("Generate Presentation", value=True)
        
        # Display agent status
        st.subheader("📊 System Status")
        
        # Check API key status
        api_keys = {
            "Gemini": bool(gemini_api_key),
            "OpenRouter": bool(openrouter_api_key),
            "Groq": bool(groq_api_key),
            "Serper": bool(serper_api_key)
        }
        
        for service, status in api_keys.items():
            status_color = "🟢" if status else "🔴"
            st.markdown(f"{status_color} **{service}**: {'Connected' if status else 'Disconnected'}")
        
        # Overall system status
        all_connected = all(api_keys.values())
        if all_connected:
            st.success("✅ All systems operational")
        else:
            st.warning("⚠️ Some services disconnected")
        
        # Help and Documentation
        st.subheader("📚 Help & Documentation")
        with st.expander("How to get API keys"):
            st.markdown("""
            - **Gemini API**: Visit [Google AI Studio](https://makersuite.google.com/)
            - **OpenRouter**: Sign up at [OpenRouter.ai](https://openrouter.ai/)
            - **Groq**: Get API keys from [Groq Console](https://console.groq.com/)
            - **Serper**: Register at [Serper.dev](https://serper.dev/)
            """)
        
        # System information
        st.subheader("ℹ️ System Info")
        st.info(f"Version: 1.0.0")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Research Setup", "Execution", "Results", "Download"])
    
    with tab1:
        st.markdown('<div class="sub-header">Research Setup</div>', unsafe_allow_html=True)
        
        # Research Topic Input
        research_topic = st.text_area(
            "Research Topic",
            placeholder="Enter your research topic or research question...",
            height=150,
            help="Be specific about your research focus for better results"
        )
        
        # Research Requirements
        st.markdown("### Research Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            research_type = st.selectbox("Research Type", ["Theoretical", "Empirical", "Experimental", "Review", "Mixed Methods"])
            deadline = st.date_input("Research Deadline")
            target_audience = st.text_input("Target Audience", placeholder="e.g., Academics, Industry Professionals")
            
        with col2:
            keywords = st.text_area("Keywords", placeholder="Enter relevant keywords separated by commas", height=100)
            data_sources = st.text_area("Preferred Data Sources", placeholder="e.g., PubMed, IEEE Xplore, Google Scholar", height=100)
        
        # Additional Requirements
        st.markdown("### Additional Requirements")
        methodology_requirements = st.text_area(
            "Methodology Requirements",
            placeholder="Describe any specific methodological requirements or constraints...",
            height=100
        )
        
        formatting_requirements = st.text_area(
            "Formatting Requirements", 
            placeholder="Describe any specific formatting requirements...",
            height=100
        )
        
        # Create paper requirements dictionary
        paper_requirements = {
            "citation_style": citation_style,
            "paper_length": paper_length,
            "target_journal": target_journal,
            "research_type": research_type,
            "deadline": str(deadline),
            "target_audience": target_audience,
            "keywords": [kw.strip() for kw in keywords.split(",") if kw.strip()],
            "data_sources": [ds.strip() for ds in data_sources.split(",") if ds.strip()],
            "methodology_requirements": methodology_requirements,
            "formatting_requirements": formatting_requirements,
            "enable_plagiarism_check": enable_plagiarism_check,
            "enable_data_analysis": enable_data_analysis,
            "enable_presentation": enable_presentation
        }
        
        # Validation
        if research_topic and st.button("🔍 Validate Research Setup", type="secondary"):
            with st.spinner("Validating research setup..."):
                # Basic validation
                if len(research_topic.strip()) < 10:
                    st.error("Research topic should be more detailed (minimum 10 characters)")
                elif not paper_requirements["keywords"]:
                    st.warning("Consider adding relevant keywords for better research results")
                else:
                    st.success("✅ Research setup looks good!")
                    
                    # Show summary
                    with st.expander("Research Summary"):
                        st.write(f"**Topic**: {research_topic[:100]}{'...' if len(research_topic) > 100 else ''}")
                        st.write(f"**Type**: {research_type}")
                        st.write(f"**Citation Style**: {citation_style}")
                        st.write(f"**Keywords**: {', '.join(paper_requirements['keywords'][:5])}")
        
        # Start Research Button
        st.markdown("---")
        if st.button("🚀 Start Research Process", type="primary", use_container_width=True):
            if not research_topic.strip():
                st.error("❌ Please enter a research topic before starting.")
            else:
                # Store in session state
                st.session_state.research_topic = research_topic
                st.session_state.paper_requirements = paper_requirements
                
                st.success(f"✅ Research setup saved! Navigate to the 'Execution' tab to begin.")
                st.rerun()
        
        # Tips for better research
        with st.expander("💡 Research Tips"):
            st.markdown("""
            **For best results:**
            - Use specific, well-defined research topics
            - Include relevant keywords from your field
            - Specify your target audience and publication venue
            - Be clear about methodology preferences
            - Set realistic deadlines
            
            **Example topics:**
            - "Impact of machine learning on healthcare diagnostics in 2020-2024"
            - "Sustainable urban planning strategies for megacities"
            - "Blockchain applications in supply chain management"
            """)

    # The rest of the tabs remain the same as in the previous implementation
    # ... (tab2, tab3, tab4 implementations)

if __name__ == "__main__":
    main()
