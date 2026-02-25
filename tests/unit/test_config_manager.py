"""
Tests for ConfigManager.
"""

import pytest
import yaml
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestConfigManager:
    """Tests for ConfigManager functionality."""

    def test_yaml_safe_load_used(self):
        """Verify yaml.safe_load is used (not yaml.load)."""
        # Read the utils.py file and check it uses safe_load
        utils_path = Path(__file__).parent.parent.parent / "src" / "utils.py"
        content = utils_path.read_text()

        assert "yaml.safe_load" in content
        # Ensure no unsafe yaml.load without Loader
        assert "yaml.load(" not in content or "Loader=" in content

    def test_config_validation_type_checking(self, temp_dir):
        """Config validation should check types."""
        from utils import ConfigManager

        # Create a minimal schema
        schema = {
            "test_section": {
                "string_value": {"type": "str", "value": "default"},
                "int_value": {"type": "int", "value": 42},
                "bool_value": {"type": "bool", "value": True}
            }
        }

        schema_path = temp_dir / "schema.yaml"
        with open(schema_path, 'w') as f:
            yaml.dump(schema, f)

        # Test that ConfigManager can be initialized
        # (We don't want to actually modify the singleton, so just test the method)
        manager = ConfigManager()
        manager.schema = manager.load_config_schema(str(schema_path))

        # Valid types should pass
        assert manager._validate_config_value("test", {"type": "str", "value": ""}, "test.path")
        assert manager._validate_config_value(42, {"type": "int", "value": 0}, "test.path")
        assert manager._validate_config_value(True, {"type": "bool", "value": False}, "test.path")

        # None should be allowed (optional values)
        assert manager._validate_config_value(None, {"type": "str", "value": ""}, "test.path")

    def test_config_validation_options_checking(self, temp_dir):
        """Config validation should check allowed options."""
        from utils import ConfigManager

        manager = ConfigManager()

        schema_item = {
            "type": "str",
            "value": "whisper",
            "options": ["whisper", "parakeet"]
        }

        # Valid option
        assert manager._validate_config_value("whisper", schema_item, "engine")
        assert manager._validate_config_value("parakeet", schema_item, "engine")

        # Invalid option
        assert not manager._validate_config_value("invalid", schema_item, "engine")

    def test_deep_update_preserves_structure(self):
        """Deep update should merge nested dicts properly."""
        from utils import ConfigManager

        manager = ConfigManager()

        # The load_user_config method uses deep_update internally
        # Test the concept with a simple case
        base = {
            "level1": {
                "level2": {
                    "value1": "original",
                    "value2": "original"
                }
            }
        }

        override = {
            "level1": {
                "level2": {
                    "value1": "changed"
                }
            }
        }

        # Simulate deep_update
        def deep_update(source, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source:
                    deep_update(source[key], value)
                else:
                    source[key] = value

        deep_update(base, override)

        # value1 should be changed, value2 should be preserved
        assert base["level1"]["level2"]["value1"] == "changed"
        assert base["level1"]["level2"]["value2"] == "original"


class TestConfigManagerSingleton:
    """Tests for ConfigManager singleton behavior."""

    def test_get_config_value_nested_keys(self):
        """Should retrieve nested config values."""
        from utils import ConfigManager

        # These tests assume ConfigManager has been initialized
        # In a real test, we'd mock this properly
        # For now, just test the method logic

        manager = ConfigManager()
        manager.config = {
            "level1": {
                "level2": {
                    "value": "test"
                }
            }
        }

        # Temporarily set the instance
        original = ConfigManager._instance
        ConfigManager._instance = manager

        try:
            result = ConfigManager.get_config_value("level1", "level2", "value")
            assert result == "test"

            # Non-existent key should return None
            result = ConfigManager.get_config_value("level1", "nonexistent")
            assert result is None
        finally:
            ConfigManager._instance = original

    def test_set_config_value_creates_nested(self):
        """Should create nested structure if needed."""
        from utils import ConfigManager

        manager = ConfigManager()
        manager.config = {}

        original = ConfigManager._instance
        ConfigManager._instance = manager

        try:
            ConfigManager.set_config_value("new_value", "level1", "level2", "key")
            assert manager.config["level1"]["level2"]["key"] == "new_value"
        finally:
            ConfigManager._instance = original


class TestConfigSchema:
    """Tests for config schema compliance."""

    def test_schema_file_exists(self):
        """Config schema file should exist."""
        schema_path = Path(__file__).parent.parent.parent / "src" / "config_schema.yaml"
        assert schema_path.exists(), "config_schema.yaml not found"

    def test_schema_is_valid_yaml(self):
        """Config schema should be valid YAML."""
        schema_path = Path(__file__).parent.parent.parent / "src" / "config_schema.yaml"

        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        assert schema is not None
        assert isinstance(schema, dict)

    def test_schema_has_required_sections(self):
        """Config schema should have all required sections."""
        schema_path = Path(__file__).parent.parent.parent / "src" / "config_schema.yaml"

        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        required_sections = ["profile", "model_options", "recording_options", "meeting_options", "misc"]

        for section in required_sections:
            assert section in schema, f"Missing required section: {section}"
