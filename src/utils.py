import yaml
import os
import re
from pathlib import Path

from paths import config_path, resource_path

class ConfigManager:
    """Manages application configuration settings."""
    _instance = None

    def __init__(self):
        """Initialize the ConfigManager instance."""
        self.config = None
        self.schema = None

    @classmethod
    def initialize(cls, schema_path=None):
        """Initialize the ConfigManager with the given schema path."""
        if cls._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            cls._instance = cls()
            cls._instance.schema = cls._instance.load_config_schema(schema_path)
            cls._instance.config = cls._instance.load_default_config()
            cls._instance.load_user_config()

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls.initialize()
        if cls._instance.config is None:  # type: ignore
            cls._instance.config = {}  # type: ignore
        return cls._instance  # type: ignore

    @classmethod
    def get_schema(cls):
        """Get the configuration schema."""
        instance = cls.get_instance()
        return instance.schema

    @classmethod
    def get_config_section(cls, *keys):
        """Get a specific section of the configuration."""
        instance = cls.get_instance()

        section = instance.config
        if not section:
            return {}
        for key in keys:
            if isinstance(section, dict) and key in section:
                section = section[key]
            else:
                return {}
        return section

    @classmethod
    def get_config_value(cls, *keys):
        """Get a specific configuration value using nested keys."""
        instance = cls.get_instance()

        value = instance.config
        if not value:
            return None
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    @classmethod
    def set_config_value(cls, value, *keys):
        """Set a specific configuration value using nested keys."""
        instance = cls.get_instance()

        config: dict = instance.config
        if not config:
            instance.config = {}
            config = instance.config
            
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif getattr(config, "get", None) is None or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]  # type: ignore
        config[keys[-1]] = value  # type: ignore

    @staticmethod
    def load_config_schema(schema_path=None):
        """Load the configuration schema from a YAML file."""
        if schema_path is None:
            schema_path = resource_path("src", "config_schema.yaml")

        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema

    def load_default_config(self):
        """Load default configuration values from the schema."""
        def extract_value(item):
            if isinstance(item, dict):
                if 'value' in item:
                    return item['value']
                else:
                    return {k: extract_value(v) for k, v in item.items()}
            return item

        config = {}
        for category, settings in self.schema.items():
            config[category] = extract_value(settings)
        return config

    def _validate_config_value(self, value, schema_item, path):
        """Validate a config value against its schema definition."""
        if not isinstance(schema_item, dict) or 'type' not in schema_item:
            # Not a leaf node, skip validation
            return True

        expected_type = schema_item['type']
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool
        }

        # Allow None for optional values
        if value is None:
            return True

        # Check type
        if expected_type in type_map:
            expected_python_type = type_map[expected_type]
            if not isinstance(value, expected_python_type):
                print(f"[!] Config validation warning: '{path}' should be {expected_type}, got {type(value).__name__}. Using default.")
                return False

        # Check if value is in allowed options
        if 'options' in schema_item and value not in schema_item['options']:
            print(f"[!] Config validation warning: '{path}' value '{value}' not in allowed options {schema_item['options']}. Using default.")
            return False

        return True

    def _validate_config_section(self, user_section, schema_section, path=""):
        """Recursively validate a config section against schema."""
        if not isinstance(schema_section, dict):
            return

        for key, schema_value in schema_section.items():
            current_path = f"{path}.{key}" if path else key

            if key not in user_section:
                continue

            user_value = user_section[key]

            # If schema_value has 'type', it's a leaf node - validate it
            if isinstance(schema_value, dict) and 'type' in schema_value:
                if not self._validate_config_value(user_value, schema_value, current_path):
                    # Reset to default value
                    user_section[key] = schema_value.get('value')
            elif isinstance(schema_value, dict) and isinstance(user_value, dict):
                # Recurse into nested sections
                self._validate_config_section(user_value, schema_value, current_path)

    def load_user_config(self, config_file=None):
        """Load user configuration and merge with default config."""
        def deep_update(source, overrides):
            for key, value in overrides.items():
                # The schema is authoritative. Retired or misspelled settings
                # are ignored instead of silently restoring removed features.
                if key not in source:
                    continue
                source_value = source[key]
                if isinstance(source_value, dict):
                    # A malformed legacy scalar must not replace an entire
                    # supported section and break later nested lookups.
                    if not isinstance(value, dict):
                        continue
                    deep_update(source_value, value)
                    continue
                if isinstance(value, dict):
                    continue
                source[key] = value

        config_file = Path(config_file) if config_file else config_path()
        if config_file.is_file():
            try:
                with open(config_file, 'r', encoding='utf-8') as file:
                    user_config = yaml.safe_load(file) or {}
                    if not isinstance(user_config, dict):
                        return
                    # Validate before merging
                    self._validate_config_section(user_config, self.schema)
                    deep_update(self.config, user_config)
            except yaml.YAMLError:
                print("Error in configuration file. Using default configuration.")

    @classmethod
    def save_config(cls, config_file=None):
        """Save the current configuration to a YAML file (atomic write with retries)."""
        instance = cls.get_instance()
        # Create user config dict matching the current config
        user_config = {}
        for section, settings in instance.config.items():
            user_config[section] = settings

        import time
        filepath = Path(config_file) if config_file else config_path()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = filepath.with_suffix('.tmp')
        
        # Write to temp file first
        with open(temp_path, 'w', encoding='utf-8') as file:
            yaml.dump(instance.config, file, default_flow_style=False)
            file.flush()
            os.fsync(file.fileno())  # Ensure it's on disk
            
        # Try to replace the original file, with retries for Windows file locks
        max_retries = 3
        retry_delay = 0.1
        for attempt in range(max_retries):
            try:
                temp_path.replace(filepath)  # Atomic rename
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # If we fail after all retries, clean up temp file and raise
                    try:
                        temp_path.unlink()
                    except:
                        pass
                    raise RuntimeError(f"Failed to save config due to file lock: {e}")

    @classmethod
    def reload_config(cls):
        """
        Reload the configuration from the file.
        """
        instance = cls.get_instance()
        instance.config = instance.load_default_config()
        instance.load_user_config()

    @classmethod
    def config_file_exists(cls):
        """Check if a valid config file exists."""
        return config_path().is_file()

    @classmethod
    def console_print(cls, message):
        """Print a message to the console if enabled in the configuration."""
        if cls._instance and cls._instance.config.get('misc', {}).get('print_to_terminal'):
            print(message)

class TextProcessor:
    """Word-preserving formatting for ElevenLabs snippet output."""

    @staticmethod
    def normalize_spacing(text):
        """Normalize whitespace and punctuation spacing without changing words."""
        text = re.sub(r'\s+', ' ', (text or '').strip())
        text = re.sub(r'\s+([,.?!])', r'\1', text)
        text = re.sub(r'([,.?!])\s*\1+', r'\1', text)
        return text

    @staticmethod
    def ensure_ending_punctuation(text):
        """Ensure non-empty text ends with sentence punctuation."""
        text = (text or '').strip()
        if text and text[-1] not in '.?!':
            text += '.'
        return text

    @classmethod
    def process(cls, transcription, add_trailing_space=False):
        """Format a transcript while preserving every spoken word."""
        if not transcription or not transcription.strip():
            return transcription

        text = cls.normalize_spacing(transcription)
        text = cls.ensure_ending_punctuation(text)
        if add_trailing_space:
            text += ' '
        return text
