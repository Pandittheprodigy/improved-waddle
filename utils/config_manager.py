# utils/config_manager.py
"""
Configuration manager for the Harvard Research Paper Publication Crew.

This module handles configuration settings, API keys, and system parameters.
"""

from typing import Dict, Any, Optional
import os
import json
from pathlib import Path

class ConfigManager:
    """Manage configuration settings for the research crew."""
    
    def __init__(self):
        self.config_file = Path("config/settings.json")
        self.default_config = self._get_default_config()
        self.config = self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            "api_settings": {
                "timeout": 30,
                "max_retries": 3,
                "rate_limit": 100
            },
            "research_settings": {
                "default_citation_style": "APA",
                "max_search_results": 20,
                "enable_plagiarism_check": True,
                "enable_data_analysis": True,
                "enable_presentation": True
            },
            "agent_settings": {
                "verbose": True,
                "memory": True,
                "max_rpm": 100,
                "process_type": "hierarchical"
            },
            "ui_settings": {
                "theme": "light",
                "auto_save": True,
                "show_progress": True
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys are present
                return self._merge_configs(self.default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self.default_config
        else:
            # Create config directory if it doesn't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            # Save default config
            self.save_config(self.default_config)
            return self.default_config
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: Dict[str, Any] = None) -> bool:
        """Save configuration to file."""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'api_settings.timeout')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set a configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        current = config
        
        try:
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the final value
            current[keys[-1]] = value
            
            # Save the updated config
            return self.save_config()
        except Exception as e:
            print(f"Error setting config value: {e}")
            return False
    
    def update_api_keys(self, api_keys: Dict[str, str]) -> bool:
        """Update API key configuration."""
        success = True
        
        for key, value in api_keys.items():
            if value:  # Only update if value is provided
                env_key = key.upper()
                os.environ[env_key] = value
                
                # Also store in config file for persistence
                config_key = f"api_keys.{key}"
                if not self.set(config_key, value):
                    success = False
        
        return success
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys from environment variables."""
        api_keys = {}
        key_names = [
            "gemini_api_key", "openrouter_api_key", 
            "groq_api_key", "serper_api_key"
        ]
        
        for key_name in key_names:
            env_key = key_name.upper()
            api_keys[key_name] = os.getenv(env_key, "")
        
        return api_keys
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate the current configuration and return validation results."""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check API keys
        api_keys = self.get_api_keys()
        missing_keys = [key for key, value in api_keys.items() if not value]
        
        if missing_keys:
            validation_results["valid"] = False
            validation_results["issues"].extend([
                f"Missing API key: {key}" for key in missing_keys
            ])
        
        # Check file paths
        if not self.config_file.parent.exists():
            validation_results["warnings"].append("Config directory does not exist")
        
        # Check agent settings
        agent_settings = self.get("agent_settings", {})
        if not agent_settings.get("verbose", False):
            validation_results["warnings"].append("Agent verbose mode is disabled")
        
        return validation_results
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        try:
            self.config = self.default_config.copy()
            return self.save_config()
        except Exception as e:
            print(f"Error resetting config: {e}")
            return False
