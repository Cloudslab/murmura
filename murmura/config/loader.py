"""Configuration loading and saving utilities."""

from pathlib import Path
from typing import Union
import yaml
import json

from murmura.config.schema import Config


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        Config object

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load based on file extension
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )

    return Config(**data)


def save_config(config: Config, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file.

    Args:
        config: Config object to save
        output_path: Output file path (.yaml, .yml, or .json)

    Raises:
        ValueError: If file format is not supported
    """
    output_path = Path(output_path)
    data = config.model_dump()

    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        elif output_path.suffix == '.json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported config format: {output_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )
