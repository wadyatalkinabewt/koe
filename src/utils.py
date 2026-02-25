import yaml
import os
import re
from pathlib import Path

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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(base_dir, 'config_schema.yaml')

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

    def load_user_config(self, config_path=os.path.join('src', 'config.yaml')):
        """Load user configuration and merge with default config."""
        def deep_update(source, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source:
                    deep_update(source[key], value)
                else:
                    source[key] = value

        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as file:
                    user_config = yaml.safe_load(file)
                    # Validate before merging
                    self._validate_config_section(user_config, self.schema)
                    deep_update(self.config, user_config)
            except yaml.YAMLError:
                print("Error in configuration file. Using default configuration.")

    @classmethod
    def save_config(cls, config_path=os.path.join('src', 'config.yaml')):
        """Save the current configuration to a YAML file (atomic write with retries)."""
        instance = cls.get_instance()
        # Create user config dict matching the current config
        user_config = {}
        for section, settings in instance.config.items():
            user_config[section] = settings

        import time
        filepath = Path(config_path)
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
        config_path = os.path.join('src', 'config.yaml')
        return os.path.isfile(config_path)

    @classmethod
    def console_print(cls, message):
        """Print a message to the console if enabled in the configuration."""
        if cls._instance and cls._instance.config['misc']['print_to_terminal']:
            print(message)

class TextProcessor:
    """Centralized utility for processing and cleaning transcribed text."""

    @staticmethod
    def apply_name_replacements(text):
        """Apply configured name spelling corrections."""
        try:
            replacements = ConfigManager.get_config_value('post_processing', 'name_replacements') or {}
            for wrong, correct in replacements.items():
                import re
                # Case-insensitive word boundary replacement
                pattern = r'\b' + re.escape(wrong) + r'\b'
                text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
        except Exception:
            pass  # Don't fail if config access fails
        return text

    @staticmethod
    def remove_prompt_leak(text):
        """Remove initial prompt if it leaked into transcription."""
        try:
            model_options = ConfigManager.get_config_section('model_options')
            initial_prompt = model_options.get('common', {}).get('initial_prompt', '')
            if initial_prompt:
                # Split prompt into lines and sentences, filter each separately
                prompt_parts = []
                for line in initial_prompt.strip().split('\n'):
                    line = line.strip()
                    if line:
                        prompt_parts.append(line)
                        # Also split on periods for sentence-level matching
                        for sentence in line.split('.'):
                            sentence = sentence.strip()
                            if len(sentence) > 10:  # Only match substantial sentences
                                prompt_parts.append(sentence)

                # Remove any prompt parts that appear in the text
                for part in prompt_parts:
                    if part in text:
                        text = text.replace(part, '')
        except Exception:
            pass  # Don't fail transcription if config access fails
        return text

    @staticmethod
    def remove_filler_words(text):
        """Remove common filler words and hallucinated continuations/outros."""
        import re

        # Filler words to remove (case insensitive)
        fillers = [
            r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\beh+\b',
            r'\bhmm+\b', r'\bmm+\b', r'\bhm+\b',
        ]

        # Whisper hallucinations - ONLY remove at end of transcription
        trailing_hallucinations = [
            # YouTube outros
            r"\s*we'?ll be right back\.?\s*$",
            r"\s*thank(s| you)( for watching)?\.?\s*$",
            r"\s*subscribe to (my|the|our) channel\.?\s*$",
            r"\s*please (like and )?subscribe\.?\s*$",
            r"\s*see you (in the )?next (one|video|time)\.?\s*$",
            r"\s*don'?t forget to (like and )?subscribe\.?\s*$",
            r"\s*like (and )?subscribe\.?\s*$",
            r"\s*hit the (like|bell|subscribe)( button)?\.?\s*$",
            # Common trailing phrases
            r"\s*(so,?\s*)?that'?s (it|all)( for (today|now))?\.?\s*$",
            r"\s*bye( bye)?\.?\s*$",
            r"\s*goodbye\.?\s*$",
            r"\s*take care\.?\s*$",
            # Incomplete trailing sentences (hallucinated continuations)
            r",?\s*and I'?ll\.?\s*$",
            r",?\s*and we'?ll\.?\s*$",
            r",?\s*and I'?m\.?\s*$",
            r",?\s*so I'?ll\.?\s*$",
            r",?\s*I'?ll see\.?\s*$",
            # Music/sound descriptions
            r"\s*\[music\]\s*$",
            r"\s*\[applause\]\s*$",
            r"\s*♪.*$",
            r"\s*\u266a.*$", # Another form of the music note from server.py
        ]
        
        for pattern in trailing_hallucinations:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Clean up resulting issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\s+([,.?!])', r'\1', text)  # Space before punctuation
        text = re.sub(r'([,.?!])\s*\1+', r'\1', text)  # Duplicate punctuation
        text = re.sub(r',\s*\.', '.', text)  # Comma followed by period
        text = re.sub(r'^\s*,\s*', '', text)  # Leading comma
        return text.strip()

    @staticmethod
    def ensure_ending_punctuation(text):
        """Ensure text ends with proper punctuation."""
        text = text.strip()
        if text and text[-1] not in '.?!':
            text += '.'
        return text

    @classmethod
    def process(cls, transcription, add_trailing_space=False):
        """Apply all post-processing steps to the transcription."""
        if not transcription or not transcription.strip():
            return transcription

        text = transcription.strip()
        text = cls.remove_prompt_leak(text)
        text = cls.remove_filler_words(text)
        text = cls.apply_name_replacements(text)
        text = cls.ensure_ending_punctuation(text)
        
        if add_trailing_space:
            text += ' '  # Trailing space for easy pasting in some apps
            
        return text
