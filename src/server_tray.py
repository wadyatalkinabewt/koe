"""
Transcription Server (Headless)

Runs the transcription server in the background without a tray icon.
Supports Whisper and Parakeet engines based on WHISPER_ENGINE env var.
Controlled via the /shutdown endpoint or by killing the process.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Server logging setup (before any other imports) ---
# Two log files:
#   logs/server.log        - RotatingFileHandler for structured app logs (INFO+)
#   logs/server_stderr.log - raw stderr redirect for CUDA/C-level crashes
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 1. Structured logging with rotation
_server_log = LOGS_DIR / "server.log"
_handler = RotatingFileHandler(_server_log, maxBytes=1_000_000, backupCount=1, encoding='utf-8')
_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
_logger = logging.getLogger('koe.server')

# 2. Raw stderr redirect for CUDA/C-level crashes
_stderr_log = LOGS_DIR / "server_stderr.log"
_stderr_file = open(_stderr_log, 'a', buffering=1, encoding='utf-8')
sys.stderr = _stderr_file

# Also redirect stdout so print() statements go to the structured log file
class _StdoutToLogger:
    """Redirect stdout print() calls to the logging system."""
    def __init__(self, logger, level=logging.INFO):
        self._logger = logger
        self._level = level
        self._buf = ''
    def write(self, msg):
        if msg and msg.strip():
            self._logger.log(self._level, msg.rstrip())
    def flush(self):
        pass

sys.stdout = _StdoutToLogger(_logger)

# Load .env file for HF_TOKEN and other settings
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Setup CUDA DLLs before any CUDA imports (PATH modification more reliable than os.add_dll_directory)
def _setup_cuda_dlls():
    try:
        import site
        paths_to_add = []
        for sp in [site.getusersitepackages()] + site.getsitepackages():
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            if os.path.exists(cudnn_bin):
                paths_to_add.append(cudnn_bin)
            if os.path.exists(cublas_bin):
                paths_to_add.append(cublas_bin)
        if paths_to_add:
            os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

_setup_cuda_dlls()

# Load model BEFORE uvicorn to avoid issues
from server import load_model, app
import uvicorn
import yaml

def get_server_port():
    """Get the configured server port."""
    port = os.environ.get("KOE_SERVER_PORT")
    if port:
        return int(port)
        
    try:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            misc = config.get("misc", {})
            config_port = misc.get("server_port")
            if config_port:
                return int(config_port)
    except:
        pass
        
    return 9876  # Default fallback


def main():
    """Run the headless server."""
    # Load model
    engine = os.environ.get("WHISPER_ENGINE", "whisper")
    model = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

    print(f"[Server] Loading {engine} engine with model {model} on {device}...")
    load_model(model, device, compute_type)
    print("[Server] Model loaded!")

    # Start server
    port = get_server_port()
    print(f"[Server] Starting on http://0.0.0.0:{port}")
    # Configure uvicorn to log to our file handler instead of stdout
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(_server_log),
                "maxBytes": 1_000_000,
                "backupCount": 1,
                "formatter": "default",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["file"], "level": "WARNING"},
            "uvicorn.error": {"handlers": ["file"], "level": "WARNING"},
            "uvicorn.access": {"handlers": ["file"], "level": "WARNING"},
        },
    }
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        log_config=log_config,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
