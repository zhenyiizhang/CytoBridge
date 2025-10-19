import yaml
from typing import Union, Dict, Any
from pathlib import Path

def load_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Loads a configuration from a name, a file path, or a dictionary.
    This version is modified to be robust during the package development phase.

    Parameters
    ----------
    config
        - If a dictionary, it is returned directly.
        - If a string or Path object that points to an existing file, it is loaded
          as a YAML file.
        - If a string that is not a file path, it is assumed to be the name of a
          built-in config (e.g., 'tigon'), which is loaded from the package's
          internal `configs` directory.

    Returns
    -------
    The loaded configuration dictionary.
    """
    if isinstance(config, dict):
        # Case 1: User passed a dictionary directly
        return config

    if not isinstance(config, (str, Path)):
        raise TypeError(f"config must be a str, Path, or dict, but got {type(config)}")

    config_path = Path(config)

    # Case 2: User passed a path to a custom config file
    if config_path.is_file():
        print(f"Loading custom config from: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Case 3: User passed a name of a built-in config
    # --- This is the core modification for development phase ---
    config_name = str(config).lower()
    
    try:
        # Use __file__ to locate the current file, then navigate to the 'configs' directory.
        # Path(__file__) -> /path/to/your/project/cytobridge/utils/config.py
        # .parent        -> /path/to/your/project/cytobridge/utils/
        # .parent        -> /path/to/your/project/cytobridge/
        # / "configs"    -> /path/to/your/project/cytobridge/configs/
        pkg_config_dir = Path(__file__).parent.parent / "configs"

        built_in_path = pkg_config_dir / f"{config_name}.yaml"

        if built_in_path.is_file():
            print(f"Loading built-in config: '{config_name}'")
            with open(built_in_path, 'r', encoding='utf-8') as f:  # 修正：用 built_in_path（指向 configs/ruot.yaml）
                return yaml.safe_load(f)
        else:
            # List all available built-in configs for a helpful error message.
            available = [
                f.stem for f in pkg_config_dir.glob('*.yaml') 
                if not f.stem.startswith('_')
            ]
            raise FileNotFoundError(
                f"Built-in config '{config_name}' not found. "
                f"Available configs: {available}"
            )
    except Exception as e:
        raise FileNotFoundError(
            "Could not locate the built-in configs directory. "
            f"Ensure the 'configs' directory is a sibling of the 'utils' directory. Error: {e}"
        )