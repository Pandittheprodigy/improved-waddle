# main.py
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
from datetime import datetime
import asyncio
import tempfile
import zipfile
import base64
from io import BytesIO

# Import CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    FileReadTool,
    FileWriteTool,
    DirectorySearchTool,
    CodeDocsSearchTool,
    SerperDevTool,
    ScrapeWebsiteTool,
)

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
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
            "groq_api_key": os.getenv("GROQ_API_KEY"),
            "serper_api_key": os.getenv("SERPER_API_KEY"),
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
            # Define all tasks
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
                tools=[AcademicSearchTool(), LiteratureReviewTool()],
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
                tools=[CodeDocsSearchTool()],
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
                tools=[FileWriteTool(), FileReadTool()],
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
                raise ValueError("Missing required API keys")
            
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
            raise
    
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
            
            # Execute the crew
            logger.info(f"Starting research execution for: {research_topic}")
            results = await crew.kickoff_async()
            
            # Generate comprehensive report
            report_generator = ReportGenerator()
            final_report = report_generator.generate_report(
                research_topic=research_topic,
                results=results,
                paper_requirements=paper_requirements
            )
            
            logger.info("Research execution completed successfully")
            return final_report
            
        except Exception as e:
            logger.error(f"Error executing research: {str(e)}")
            raise

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
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">Harvard Research Paper Publication Crew</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Inputs
        st.subheader("API Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
        groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        serper_api_key = st.text_input("Serper API Key", type="password", value=os.getenv("SERPER_API_KEY", ""))
        
        # Save API keys to environment
        if st.button("Save API Keys"):
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
            os.environ["GROQ_API_KEY"] = groq_api_key
            os.environ["SERPER_API_KEY"] = serper_api_key
            st.success("API keys saved successfully!")
        
        # Research Configuration
        st.subheader("Research Configuration")
        citation_style = st.selectbox("Citation Style", ["APA", "MLA", "Chicago", "Harvard", "IEEE"])
        paper_length = st.selectbox("Paper Length", ["Short (5-10 pages)", "Medium (10-20 pages)", "Long (20+ pages)"])
        target_journal = st.text_input("Target Journal/Conference", placeholder="e.g., Nature, Science, IEEE Transactions")
        
        # Advanced Options
        st.subheader("Advanced Options")
        enable_plagiarism_check = st.checkbox("Enable Plagiarism Check", value=True)
        enable_data_analysis = st.checkbox("Enable Data Analysis", value=True)
        enable_presentation = st.checkbox("Generate Presentation", value=True)
        
        # Display agent status
        st.subheader("System Status")
        st.info("✅ All agents ready")
        st.info("✅ Tools initialized")
        st.info("✅ APIs connected")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Research Setup", "Execution", "Results", "Download"])
    
    with tab1:
        st.markdown('<div class="sub-header">Research Setup</div>', unsafe_allow_html=True)
        
        # Research Topic Input
        research_topic = st.text_area(
            "Research Topic",
            placeholder="Enter your research topic or research question...",
            height=150
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
        
        # Start Research Button
        if st.button("🚀 Start Research Process", type="primary", use_container_width=True):
            if not research_topic.strip():
                st.error("Please enter a research topic before starting.")
            else:
                # Store in session state
                st.session_state.research_topic = research_topic
                st.session_state.paper_requirements = paper_requirements
                
                # Navigate to execution tab
                st.rerun()
    
    with tab2:
        st.markdown('<div class="sub-header">Research Execution</div>', unsafe_allow_html=True)
        
        if 'research_topic' not in st.session_state:
            st.info("Please set up your research in the 'Research Setup' tab first.")
            st.stop()
        
        research_topic = st.session_state.research_topic
        paper_requirements = st.session_state.paper_requirements
        
        st.markdown(f"**Research Topic:** {research_topic}")
        st.markdown(f"**Research Type:** {paper_requirements['research_type']}")
        st.markdown(f"**Target Journal:** {paper_requirements['target_journal']}")
        
        # Execution Progress
        st.markdown("### Execution Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Agent execution status
        agent_status = st.empty()
        
        # Start execution
        if st.button("Execute Research Crew", type="primary", use_container_width=True):
            try:
                # Initialize Harvard Research Crew
                harvard_crew = HarvardResearchCrew()
                
                # Update progress
                status_text.text("Initializing agents...")
                progress_bar.progress(10)
                
                # Create and execute crew
                with st.spinner("Creating research crew and starting execution..."):
                    results = asyncio.run(
                        harvard_crew.execute_research(research_topic, paper_requirements)
                    )
                
                # Update session state with results
                st.session_state.execution_results = results
                st.session_state.crew = harvard_crew
                
                st.success("Research execution completed successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
                logger.error(f"Execution error: {str(e)}")
    
    with tab3:
        st.markdown('<div class="sub-header">Research Results</div>', unsafe_allow_html=True)
        
        if 'execution_results' not in st.session_state:
            st.info("Please execute the research process in the 'Execution' tab first.")
            st.stop()
        
        results = st.session_state.execution_results
        
        # Display results summary
        st.markdown("### Research Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Research Topic", results.get("topic", "N/A"))
        
        with col2:
            st.metric("Research Type", results.get("research_type", "N/A"))
        
        with col3:
            st.metric("Execution Time", results.get("execution_time", "N/A"))
        
        # Detailed results
        st.markdown("### Detailed Results")
        
        # Literature Review Results
        if "literature_review" in results:
            with st.expander("Literature Review Results", expanded=True):
                st.markdown("#### Key Findings")
                st.write(results["literature_review"].get("key_findings", "N/A"))
                
                st.markdown("#### Research Gaps Identified")
                st.write(results["literature_review"].get("research_gaps", "N/A"))
        
        # Data Analysis Results
        if "data_analysis" in results and paper_requirements.get("enable_data_analysis", True):
            with st.expander("Data Analysis Results"):
                st.markdown("#### Analysis Summary")
                st.write(results["data_analysis"].get("summary", "N/A"))
                
                # Display any generated charts or visualizations
                if "visualizations" in results["data_analysis"]:
                    for i, viz in enumerate(results["data_analysis"]["visualizations"]):
                        st.image(viz, caption=f"Visualization {i+1}")
        
        # Research Paper Results
        if "research_paper" in results:
            with st.expander("Research Paper", expanded=True):
                st.markdown("#### Paper Structure")
                for section in results["research_paper"].get("sections", []):
                    st.markdown(f"- **{section['title']}**")
                
                st.markdown("#### Download Paper")
                paper_content = results["research_paper"].get("content", "")
                if paper_content:
                    st.download_button(
                        label="Download Research Paper (PDF)",
                        data=paper_content,
                        file_name=f"{research_topic.replace(' ', '_')}_research_paper.pdf",
                        mime="application/pdf"
                    )
        
        # Presentation Results
        if "presentation" in results and paper_requirements.get("enable_presentation", True):
            with st.expander("PowerPoint Presentation"):
                st.markdown("#### Presentation Overview")
                st.write(f"Number of slides: {results['presentation'].get('slide_count', 'N/A')}")
                
                st.markdown("#### Download Presentation")
                presentation_content = results["presentation"].get("content", "")
                if presentation_content:
                    st.download_button(
                        label="Download Presentation (PPTX)",
                        data=presentation_content,
                        file_name=f"{research_topic.replace(' ', '_')}_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
        
        # Quality Assurance Results
        if "quality_assurance" in results:
            with st.expander("Quality Assurance Report"):
                qa_report = results["quality_assurance"]
                
                st.markdown("#### Content Accuracy")
                st.write(qa_report.get("content_accuracy", "N/A"))
                
                st.markdown("#### Structure Evaluation")
                st.write(qa_report.get("structure_evaluation", "N/A"))
                
                st.markdown("#### Recommendations")
                st.write(qa_report.get("recommendations", "N/A"))
    
    with tab4:
        st.markdown('<div class="sub-header">Download Results</div>', unsafe_allow_html=True)
        
        if 'execution_results' not in st.session_state:
            st.info("Please execute the research process first to generate downloadable results.")
            st.stop()
        
        results = st.session_state.execution_results
        
        st.markdown("### Download All Results")
        
        # Create a zip file containing all results
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add research paper
            if "research_paper" in results:
                paper_content = results["research_paper"].get("content", "")
                if paper_content:
                    zip_file.writestr("research_paper.pdf", paper_content)
            
            # Add presentation
            if "presentation" in results:
                presentation_content = results["presentation"].get("content", "")
                if presentation_content:
                    zip_file.writestr("presentation.pptx", presentation_content)
            
            # Add literature review
            if "literature_review" in results:
                lr_content = json.dumps(results["literature_review"], indent=2)
                zip_file.writestr("literature_review.json", lr_content)
            
            # Add data analysis results
            if "data_analysis" in results:
                da_content = json.dumps(results["data_analysis"], indent=2)
                zip_file.writestr("data_analysis.json", da_content)
            
            # Add quality assurance report
            if "quality_assurance" in results:
                qa_content = json.dumps(results["quality_assurance"], indent=2)
                zip_file.writestr("quality_assurance.json", qa_content)
            
            # Add execution summary
            summary_content = json.dumps(results, indent=2)
            zip_file.writestr("execution_summary.json", summary_content)
        
        zip_buffer.seek(0)
        
        # Download button for all results
        st.download_button(
            label="Download All Results (ZIP)",
            data=zip_buffer,
            file_name=f"{research_topic.replace(' ', '_')}_all_results.zip",
            mime="application/zip"
        )
        
        # Individual download buttons
        st.markdown("### Individual Downloads")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "research_paper" in results:
                paper_content = results["research_paper"].get("content", "")
                if paper_content:
                    st.download_button(
                        label="Download Research Paper",
                        data=paper_content,
                        file_name=f"{research_topic.replace(' ', '_')}_research_paper.pdf",
                        mime="application/pdf"
                    )
        
        with col2:
            if "presentation" in results:
                presentation_content = results["presentation"].get("content", "")
                if presentation_content:
                    st.download_button(
                        label="Download Presentation",
                        data=presentation_content,
                        file_name=f"{research_topic.replace(' ', '_')}_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
        
        with col3:
            if "quality_assurance" in results:
                qa_content = json.dumps(results["quality_assurance"], indent=2)
                st.download_button(
                    label="Download QA Report",
                    data=qa_content,
                    file_name="quality_assurance_report.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
