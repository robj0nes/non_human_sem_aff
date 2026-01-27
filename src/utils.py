import datetime
import logging
import pickle
import platform

import torch
import yaml

from pathlib import Path
from typing import Union, Dict, Any

from groundingdino.util.inference import load_model

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    """
    path = Path(config_path)
    if not path.exists():
        # Logging here ensures the error is recorded even if the traceback is suppressed later
        logging.error(f"Config file not found at: {path}")
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_pickle(path: Path) -> Any:
    """Safely loads a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Safely saves data to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_dino_model(config):
    # Determine device
    if platform.system() == 'Darwin':
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    torch.set_default_device(device)
    logging.info(f'Device set to: {device} (CUDA Avail: {torch.cuda.is_available()})')

    gd_config_path = ROOT / config['paths']['grounding_dino_config']
    gd_weights_path = ROOT / config['paths']['grounding_dino_weights']
    return load_model(str(gd_config_path), str(gd_weights_path))


class Log:
    def __init__(self, path):
        time_stamp = datetime.datetime.now().strftime("%H_%M")
        self.path = Path(path / f"{time_stamp}_log.txt")
        open(Path(self.path), 'w').close()  # Make an empty log file.
        self.cursor = ""

    def _tab_cursor(self):
        self.cursor += "    "

    def append(self, msg, tab_after=False):
        with open(self.path, 'a') as f:
            f.write(self.cursor + msg + '\n')
        if tab_after:
            self._tab_cursor()

    def untab_cursor(self, num_tabs):
        for n in range(num_tabs):
            self.cursor = self.cursor[:-4]

    def reset_cursor(self):
        self.cursor = ""

