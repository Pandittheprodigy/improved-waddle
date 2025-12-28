# tools/presentation_tools.py
"""
Tools for creating professional presentations in the Harvard Research Paper Publication Crew.

This module contains specialized tools for PowerPoint presentation creation,
visual design, and data visualization.
"""

from typing import Dict, List, Any, Optional
from crewai_tools import BaseTool
import io
import base64
from datetime import datetime

class PowerPointPresentationTool(BaseTool):
    """Tool for creating professional PowerPoint presentations."""
    
    name: str = "PowerPoint Presentation Tool"
    description: str = "Create professional PowerPoint presentations with custom slides, layouts, and content."
    
    def _run(
        self, 
        presentation_data: Dict[str, Any], 
        template_style: str = "Professional",
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Create a PowerPoint presentation from research data.
        
        Args:
            presentation_data: Dictionary containing presentation content and structure
            template_style: Style template for the presentation
            include_visualizations: Whether to include data visualizations
            
        Returns:
            Dictionary containing presentation details and download information
        """
        # Validate input data
        required_fields = ["title", "slides"]
        for field in required_fields:
            if field not in presentation_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Generate presentation
        presentation_content = self._generate_presentation(
            presentation_data, 
            template_style, 
            include_visualizations
        )
        
        # Calculate presentation metrics
        slide_count = len(presentation_data["slides"])
        estimated_duration = self._estimate_presentation_duration(slide_count)
        
        return {
            "title": presentation_data["title"],
            "slide_count": slide_count,
            "template_style": template_style,
            "estimated_duration": estimated_duration,
            "content_generated": presentation_content,
            "includes_visualizations": include_visualizations,
            "timestamp": datetime.now().isoformat(),
            "file_format": "PPTX",
            "file_size": len(presentation_content)  # Mock file size
        }
    
    def _generate_presentation(
        self, 
        data: Dict[str, Any], 
        style: str, 
        include_viz: bool
    ) -> bytes:
        """Generate the actual PowerPoint presentation content."""
        # Mock implementation - in reality, you would use python-pptx or similar library
        # This returns mock binary content representing a PPTX file
        
        # Create mock PPTX content
        mock_content = f"""
        PowerPoint Presentation: {data['title']}
        Style: {style}
        Slides: {len(data['slides'])}
        
        Slide Content:
        """
        
        for i, slide in enumerate(data["slides"], 1):
            mock_content += f"\nSlide {i}: {slide.get('title', 'Untitled')}\n"
            mock_content += f"Content: {slide.get('content', 'No content')}\n"
            
            if include_viz and "visualizations" in slide:
                mock_content += f"Visualizations: {len(slide['visualizations'])}\n"
        
        # Convert to bytes (mock PPTX content)
        return mock_content.encode('utf-8')
    
    def _estimate_presentation_duration(self, slide_count: int) -> str:
        """Estimate the duration of the presentation based on slide count."""
        # Average 2 minutes per slide
        minutes = slide_count * 2
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if hours > 0:
            return f"{hours} hour(s) and {remaining_minutes} minute(s)"
        else:
            return f"{minutes} minutes"

class VisualDesignTool(BaseTool):
    """Tool for creating visually appealing presentation designs."""
    
    name: str = "Visual Design Tool"
    description: str = "Create visually appealing designs, layouts, and visual elements for presentations."
    
    def _run(
        self, 
        design_requirements: Dict[str, Any], 
        color_scheme: str = "Professional Blue",
        font_style: str = "Clean Sans"
    ) -> Dict[str, Any]:
        """
        Create visual design elements for presentations.
        
        Args:
            design_requirements: Dictionary specifying design requirements
            color_scheme: Color scheme for the design
            font_style: Font style for text elements
            
        Returns:
            Dictionary containing design specifications and assets
        """
        # Validate design requirements
        required_elements = design_requirements.get("required_elements", [])
        
        # Generate design specifications
        design_specifications = self._generate_design_specifications(
            required_elements, 
            color_scheme, 
            font_style
        )
        
        # Create mock design assets
        design_assets = self._create_design_assets(design_specifications)
        
        return {
            "color_scheme": color_scheme,
            "font_style": font_style,
            "required_elements": required_elements,
            "design_specifications": design_specifications,
            "design_assets": design_assets,
            "compliance_check": self._check_design_compliance(design_specifications),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_design_specifications(
        self, 
        elements: List[str], 
        color_scheme: str, 
        font_style: str
    ) -> Dict[str, Any]:
        """Generate detailed design specifications."""
        specifications = {
            "color_palette": self._get_color_palette(color_scheme),
            "typography": {
                "headings": f"{font_style} Bold, 32pt",
                "body_text": f"{font_style} Regular, 18pt", 
                "captions": f"{font_style} Light, 14pt"
            },
            "layout_grids": {
                "title_slide": "Full-width title with subtitle",
                "content_slide": "Two-column layout with image option",
                "data_slide": "Chart-focused with supporting text"
            },
            "design_elements": elements,
            "spacing": {
                "margins": "1 inch",
                "line_height": "1.2",
                "element_spacing": "0.5 inch"
            }
        }
        
        return specifications
    
    def _get_color_palette(self, scheme: str) -> Dict[str, str]:
        """Get color palette for the specified scheme."""
        palettes = {
            "Professional Blue": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e", 
                "accent": "#2ca02c",
                "background": "#ffffff",
                "text": "#333333"
            },
            "Corporate Grey": {
                "primary": "#444444",
                "secondary": "#666666",
                "accent": "#007acc", 
                "background": "#f5f5f5",
                "text": "#000000"
            },
            "Academic Green": {
                "primary": "#2e7d32",
                "secondary": "#81c784",
                "accent": "#ffd54f",
                "background": "#ffffff", 
                "text": "#333333"
            }
        }
        
        return palettes.get(scheme, palettes["Professional Blue"])
    
    def _create_design_assets(self, specifications: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create design assets based on specifications."""
        assets = []
        
        # Create mock assets
        asset_types = ["icons", "charts", "infographics", "templates"]
        
        for asset_type in asset_types:
            assets.append({
                "type": asset_type,
                "description": f"Professional {asset_type} matching the design specifications",
                "color_scheme": specifications["color_palette"],
                "dimensions": "Varies by usage",
                "format": "Vector/SVG"
            })
        
        return assets
    
    def _check_design_compliance(self, specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Check if design meets accessibility and professional standards."""
        compliance_issues = []
        
        # Check color contrast
        primary_color = specifications["color_palette"]["primary"]
        text_color = specifications["color_palette"]["text"]
        
        # Mock contrast check
        contrast_ratio = 4.5  # Mock value
        
        if contrast_ratio < 4.5:
            compliance_issues.append("Insufficient color contrast for accessibility")
        
        # Check font sizes
        body_font_size = int(specifications["typography"]["body_text"].split(",")[1].strip().replace("pt", ""))
        if body_font_size < 14:
            compliance_issues.append("Body text font size too small for presentations")
        
        return {
            "overall_compliance": len(compliance_issues) == 0,
            "issues": compliance_issues,
            "recommendations": self._get_compliance_recommendations(compliance_issues)
        }
    
    def _get_compliance_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations to address compliance issues."""
        recommendations = []
        
        for issue in issues:
            if "contrast" in issue.lower():
                recommendations.append("Increase color contrast ratio to at least 4.5:1")
            elif "font size" in issue.lower():
                recommendations.append("Increase body text font size to at least 18pt")
        
        return recommendations if recommendations else ["Design meets all compliance standards"]

class DataVisualizationTool(BaseTool):
    """Tool for creating data visualizations and charts."""
    
    name: str = "Data Visualization Tool"
    description: str = "Create professional data visualizations, charts, and graphs for presentations and reports."
    
    def _run(
        self, 
        data: Dict[str, Any], 
        chart_type: str = "bar",
        customization: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create data visualizations from provided data.
        
        Args:
            data: Dictionary containing data to visualize
            chart_type: Type of chart to create (bar, line, pie, scatter, etc.)
            customization: Customization options for the chart
            
        Returns:
            Dictionary containing visualization details and assets
        """
        if not data:
            raise ValueError("No data provided for visualization")
        
        # Generate visualization
        visualization_result = self._create_visualization(data, chart_type, customization)
        
        return {
            "chart_type": chart_type,
            "data_source": data.get("source", "Research analysis"),
            "visualization_details": visualization_result,
            "customization_applied": customization or {},
            "accessibility_features": self._check_accessibility(chart_type),
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_visualization(
        self, 
        data: Dict[str, Any], 
        chart_type: str, 
        customization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create the actual visualization."""
        # Mock implementation - in reality, you would use matplotlib, plotly, or similar
        
        chart_details = {
            "title": data.get("title", f"{chart_type.capitalize()} Chart"),
            "description": data.get("description", f"Visualization of {chart_type} data"),
            "data_points": len(data.get("values", [])),
            "chart_style": customization.get("style", "default") if customization else "default",
            "color_scheme": customization.get("colors", "auto") if customization else "auto"
        }
        
        # Mock chart generation
        chart_image = self._generate_chart_image(chart_type, data)
        
        return {
            "chart_details": chart_details,
            "image_data": chart_image,
            "file_format": "PNG",
            "dimensions": "800x600 pixels",
            "interactive": False
        }
    
    def _generate_chart_image(self, chart_type: str, data: Dict[str, Any]) -> str:
        """Generate mock chart image data."""
        # In a real implementation, you would create actual chart images
        # This returns a mock base64-encoded image
        
        mock_image_data = f"Mock {chart_type} chart image for data: {data.get('title', 'Untitled')}"
        return base64.b64encode(mock_image_data.encode()).decode()
    
    def _check_accessibility(self, chart_type: str) -> Dict[str, Any]:
        """Check if the visualization meets accessibility standards."""
        accessibility_features = {
            "alt_text": f"{chart_type.capitalize()} chart visualization",
            "color_blind_friendly": True,
            "data_labels": True,
            "legend_included": True,
            "high_contrast": True
        }
        
        # Specific checks based on chart type
        if chart_type.lower() == "pie":
            accessibility_features["data_labels"] = True
            accessibility_features["legend_included"] = True
        
        return accessibility_features
