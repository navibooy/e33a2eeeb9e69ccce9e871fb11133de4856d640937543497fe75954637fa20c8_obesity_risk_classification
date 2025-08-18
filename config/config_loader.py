"""
Configuration loader for YAML-based configuration.
Provides a clean interface to access configuration values.
"""

import os
import yaml
from typing import Dict, Any, List

class ConfigLoader:
    """
    Configuration loader that handles YAML config files with environment overrides.
    """

    def __init__(self, config_path: str = None, environment: str = "production"):
        """
        Initialize the configuration loader.

        Args:
            config_path (str): Path to the YAML config file
            environment (str): Environment to load (development, production, testing)
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "preprocessing_config.yaml")

        self.config_path = config_path
        self.environment = environment
        self._config = self._load_config()
        self._apply_environment_overrides()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        if 'environments' in self._config and self.environment in self._config['environments']:
            overrides = self._config['environments'][self.environment]
            self._deep_update(self._config, overrides)
            print(f"Applied {self.environment} environment overrides")

    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path (str): Dot-separated path to config value (e.g., 'dataset.paths.raw_data')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")

    def get_paths(self) -> Dict[str, str]:
        """Get all file paths."""
        return self.get('dataset.paths')

    def get_target_config(self) -> Dict[str, Any]:
        """Get target column configuration."""
        return self.get('dataset.target')

    def get_numerical_features(self) -> List[str]:
        """Get list of numerical features."""
        return self.get('features.numerical', [])

    def get_categorical_features(self) -> List[str]:
        """Get list of categorical features."""
        return self.get('features.categorical', [])

    def get_all_features(self) -> List[str]:
        """Get all features (numerical + categorical)."""
        return self.get_numerical_features() + self.get_categorical_features()

    def get_column_mapping(self) -> Dict[str, str]:
        """Get column renaming mapping."""
        return self.get('features.column_mapping', {})

    def get_exclude_columns(self) -> List[str]:
        """Get columns to exclude from features."""
        target_config = self.get_target_config()
        return [target_config['target_column']] + target_config['id_columns']

    def get_missing_value_config(self) -> Dict[str, Any]:
        """Get missing value handling configuration."""
        return self.get('data_cleaning.missing_values')

    def get_obesity_mapping(self) -> Dict[str, str]:
        """Get obesity type mapping."""
        return self.get('data_cleaning.obesity_type_mapping')

    def get_bmi_config(self) -> Dict[str, str]:
        """Get BMI calculation configuration."""
        return self.get('feature_engineering.bmi')

    def get_body_fat_config(self) -> Dict[str, Any]:
        """Get body fat calculation configuration."""
        return self.get('feature_engineering.body_fat')

    def get_split_config(self) -> Dict[str, Any]:
        """Get train/test split configuration."""
        return self.get('train_test_split')

    def get_drift_config(self) -> Dict[str, Any]:
        """Get drift generation configuration."""
        return self.get('drift')

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.get('validation')

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging')

    def validate_config(self):
        """Validate configuration consistency."""
        try:
            required_sections = ['dataset', 'features', 'data_cleaning', 'train_test_split', 'drift']
            for section in required_sections:
                if section not in self._config:
                    raise ValueError(f"Required configuration section missing: {section}")

            numerical = set(self.get_numerical_features())
            categorical = set(self.get_categorical_features())
            overlap = numerical.intersection(categorical)
            if overlap:
                raise ValueError(f"Features cannot be both numerical and categorical: {overlap}")

            target_col = self.get('dataset.target.target_column')
            all_features = set(self.get_all_features())
            if target_col in all_features:
                raise ValueError(f"Target column '{target_col}' cannot be in feature lists")

            print("Configuration validation passed")

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            raise

    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)

        print(f"Environment: {self.environment}")
        print(f"Config file: {self.config_path}")

        print("\nDataset:")
        print(f"  Target column: {self.get('dataset.target.target_column')}")
        print(f"  Raw data path: {self.get('dataset.paths.raw_data')}")

        print("\nFeatures:")
        print(f"  Numerical ({len(self.get_numerical_features())}): {self.get_numerical_features()}")
        print(f"  Categorical ({len(self.get_categorical_features())}): {self.get_categorical_features()}")

        print("\nSplit configuration:")
        split_config = self.get_split_config()
        print(f"  Test size: {split_config['test_size']}")
        print(f"  Random state: {split_config['random_state']}")
        print(f"  Stratify: {split_config['stratify']}")

        print("\nDrift configuration:")
        drift_config = self.get_drift_config()
        print(f"  Numerical train method: {drift_config['numerical']['train']['method']}")
        print(f"  Categorical train flip: {drift_config['categorical']['train']['flip_percentage']*100}%")

# Global config instance
_config_instance = None

def get_config(config_path: str = None, environment: str = "production") -> ConfigLoader:
    """
    Get global configuration instance (singleton).

    Args:
        config_path (str): Path to config file (only used on first call)
        environment (str): Environment name (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path, environment)
    return _config_instance

def reload_config(config_path: str = None, environment: str = "production") -> ConfigLoader:
    """
    Force reload of configuration.

    Args:
        config_path (str): Path to config file
        environment (str): Environment name

    Returns:
        New ConfigLoader instance
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path, environment)
    return _config_instance

# Functions for common access patterns
def get_numerical_features() -> List[str]:
    """Get numerical features from global config."""
    return get_config().get_numerical_features()

def get_categorical_features() -> List[str]:
    """Get categorical features from global config."""
    return get_config().get_categorical_features()

def get_paths() -> Dict[str, str]:
    """Get file paths from global config."""
    return get_config().get_paths()

def get_target_column() -> str:
    """Get target column name from global config."""
    return get_config().get('dataset.target.target_column')

if __name__ == "__main__":
    config = ConfigLoader(environment="development")
    config.validate_config()
    config.print_summary()
