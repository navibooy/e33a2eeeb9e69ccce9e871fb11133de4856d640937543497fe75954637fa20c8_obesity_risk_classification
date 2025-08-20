# config/config_loader.py
"""
Configuration loader for YAML-based configuration.
Provides a clean interface to access configuration values.
"""

import os
import yaml
import socket  # Added for Docker detection
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
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        self.config_path = config_path
        self.environment = environment
        self._config = self._load_config()
        self._apply_environment_overrides()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"✓ Configuration loaded from {self.config_path}")
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
            print(f"✓ Applied {self.environment} environment overrides")

    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    # def _detect_docker_environment(self) -> bool:
    #     """Detect if running in Docker/Airflow environment."""
    #     return (
    #         os.path.exists('/.dockerenv') or           # Docker container indicator
    #         os.environ.get('AIRFLOW_HOME') or          # Airflow environment
    #         os.path.exists('/opt/airflow') or          # Airflow working directory
    #         self._can_resolve_mlflow_service()         # Can resolve mlflow hostname
    #     )

    # def _can_resolve_mlflow_service(self) -> bool:
    #     """Check if 'mlflow' hostname can be resolved (Docker environment)."""
    #     try:
    #         socket.gethostbyname('mlflow')
    #         return True
    #     except socket.gaierror:
    #         return False

    def _is_docker_compose_running(self) -> bool:
        """Simple check if Docker Compose is running."""
        try:
            # Check if MLflow service is accessible via Docker service name
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 second timeout
            result = sock.connect_ex(('mlflow', 5000))
            sock.close()
            return result == 0
        except Exception:
            return False

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

    def get_model_training_config(self) -> Dict[str, Any]:
        """Get model training configuration."""
        return self.get('model_training')

    def get_mlflow_config(self) -> Dict[str, Any]:
        """
        Get MLFlow configuration with simple Docker Compose detection.
        Uses mlflow:5000 when Docker Compose is up, localhost:5000 otherwise.
        """
        # Start with base config from YAML
        base_config = self.get('model_training.mlflow', {})

        # Check if environment variable is set (highest priority)
        env_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if env_uri:
            tracking_uri = env_uri
            print(f"✓ Using MLflow URI from environment: {env_uri}")
        else:
            # Auto-detect based on Docker Compose status
            if self._is_docker_compose_running():
                tracking_uri = "http://mlflow:5000"
                print("✓ Docker Compose detected - using mlflow:5000")
            else:
                tracking_uri = "http://localhost:5000"
                print("✓ Docker Compose not running - using localhost:5000")

        # Build MLflow config
        mlflow_config = {
            'tracking_uri': tracking_uri,
            'experiment_name': os.environ.get('MLFLOW_EXPERIMENT_NAME',
                                            base_config.get('experiment_name', 'obesity_risk_classification')),
            'backend_store_uri': base_config.get('backend_store_uri', 'mlflow/runs'),
            'default_artifact_root': base_config.get('default_artifact_root', '/mlflow/artifacts')
        }

        return mlflow_config
    # def get_mlflow_config(self) -> Dict[str, Any]:
    #     """
    #     Get MLFlow configuration with Docker environment detection.
    #     ✅ NEW: Docker-aware MLflow configuration
    #     """
    #     # Start with base config from YAML
    #     base_config = self.get('model_training.mlflow', {})

    #     # Method 1: Check if MLflow URI is set by Airflow DAG (highest priority)
    #     env_uri = os.environ.get('MLFLOW_TRACKING_URI')
    #     if env_uri:
    #         print(f"✓ Using MLflow URI from environment variable: {env_uri}")
    #         tracking_uri = env_uri
    #     else:
    #         # Method 2: Detect environment and set appropriate URI
    #         if self._detect_docker_environment():
    #             tracking_uri = "http://mlflow:5000"
    #             print("✓ Docker environment detected, using MLflow service name")
    #         else:
    #             # Use config file value or default to localhost
    #             tracking_uri = base_config.get('tracking_uri', 'http://localhost:5000')
    #             print("✓ Local environment detected, using localhost")

    #     # Build enhanced MLflow config
    #     mlflow_config = {
    #         'tracking_uri': tracking_uri,
    #         'experiment_name': os.environ.get('MLFLOW_EXPERIMENT_NAME',
    #                                         base_config.get('experiment_name', 'obesity_risk_classification')),
    #         'backend_store_uri': os.environ.get('MLFLOW_BACKEND_STORE_URI',
    #                                           base_config.get('backend_store_uri')),
    #         'default_artifact_root': os.environ.get('MLFLOW_DEFAULT_ARTIFACT_ROOT',
    #                                                base_config.get('default_artifact_root', '/mlflow/artifacts')),
    #         # Preserve any other config from YAML
    #         **{k: v for k, v in base_config.items() if k not in ['tracking_uri', 'experiment_name']}
    #     }

    #     print(f"✓ MLflow config resolved: tracking_uri={mlflow_config['tracking_uri']}, experiment={mlflow_config['experiment_name']}")
    #     return mlflow_config

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.get('model_training.hyperparameters')

    def get_model_features_config(self) -> Dict[str, Any]:
        """Get model feature configuration."""
        return self.get('model_training.features')

    def get_artifacts_config(self) -> Dict[str, Any]:
        """Get model artifacts configuration."""
        return self.get('model_training.artifacts')

    def validate_config(self):
        """Validate configuration consistency."""
        try:
            # Check required sections exist
            required_sections = ['dataset', 'features', 'data_cleaning', 'train_test_split', 'drift']
            for section in required_sections:
                if section not in self._config:
                    raise ValueError(f"Required configuration section missing: {section}")

            # Check feature overlap
            numerical = set(self.get_numerical_features())
            categorical = set(self.get_categorical_features())
            overlap = numerical.intersection(categorical)
            if overlap:
                raise ValueError(f"Features cannot be both numerical and categorical: {overlap}")

            # Check target column not in features
            target_col = self.get('dataset.target.target_column')
            all_features = set(self.get_all_features())
            if target_col in all_features:
                raise ValueError(f"Target column '{target_col}' cannot be in feature lists")

            print("✓ Configuration validation passed")

        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
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

        print("\nModel training configuration:")
        model_config = self.get_model_training_config()
        print(f"  MLFlow URI: {model_config['mlflow']['tracking_uri']}")
        print(f"  Hyperparameters: {model_config['hyperparameters']}")
        print(f"  Model save path: {model_config['artifacts']['model_save_path']}")

# Global config instance (singleton pattern)
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

# Convenience functions for common access patterns
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

# Example usage
if __name__ == "__main__":
    # Test the configuration loader
    config = ConfigLoader(environment="development")
    config.validate_config()
    config.print_summary()
