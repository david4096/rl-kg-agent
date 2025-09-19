"""Environment configuration utilities for RL-KG-Agent."""

import os
import logging
from pathlib import Path
from typing import Optional

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_environment_config(env_file: Optional[str] = None) -> None:
    """Load environment configuration from .env file and set SSL certificates.
    
    Args:
        env_file: Path to specific .env file. If None, looks for .env in project root.
    """
    if DOTENV_AVAILABLE:
        # Find project root
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            if (current_dir / "pyproject.toml").exists():
                project_root = current_dir
                break
            current_dir = current_dir.parent
        else:
            project_root = Path.cwd()
        
        # Load .env file
        if env_file:
            env_path = Path(env_file)
        else:
            env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment configuration from {env_path}")
        else:
            logger.warning(f"No .env file found at {env_path}")
    
    # Configure SSL certificates
    setup_ssl_config()


def setup_ssl_config() -> None:
    """Setup SSL certificate configuration for various libraries."""
    ssl_cert_file = os.getenv("SSL_CERT_FILE")
    
    if ssl_cert_file:
        # Expand ~ in path
        ssl_cert_file = os.path.expanduser(ssl_cert_file)
        
        if os.path.exists(ssl_cert_file):
            # Set environment variables for various libraries
            os.environ["SSL_CERT_FILE"] = ssl_cert_file
            os.environ["REQUESTS_CA_BUNDLE"] = ssl_cert_file
            os.environ["CURL_CA_BUNDLE"] = ssl_cert_file
            
            # For Python SSL
            os.environ["PYTHONHTTPSVERIFY"] = "1"
            
            # Disable tokenizer parallelism to avoid fork warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            logger.info(f"✅ SSL certificate configured: {ssl_cert_file}")
            logger.info("✅ Environment variables exported: SSL_CERT_FILE, REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE")
        else:
            logger.warning(f"SSL certificate file not found: {ssl_cert_file}")
    else:
        logger.info("No SSL_CERT_FILE specified in environment")


def get_cache_directories() -> dict:
    """Get configured cache directories for various libraries."""
    cache_dirs = {}
    
    # Sentence Transformers
    st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if st_home:
        cache_dirs["sentence_transformers"] = os.path.expanduser(st_home)
    
    # PyTorch Hub
    torch_home = os.getenv("TORCH_HOME")
    if torch_home:
        cache_dirs["torch"] = os.path.expanduser(torch_home)
    
    # HuggingFace
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        cache_dirs["huggingface"] = os.path.expanduser(hf_home)
    
    return cache_dirs


def ensure_cache_directories() -> None:
    """Ensure cache directories exist."""
    cache_dirs = get_cache_directories()
    
    for name, path in cache_dirs.items():
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory ready: {name} -> {path}")
        except Exception as e:
            logger.warning(f"Could not create cache directory {name} at {path}: {e}")


# Initialize environment on module import
def init_environment():
    """Initialize environment configuration."""
    try:
        load_environment_config()
        ensure_cache_directories()
    except Exception as e:
        logger.warning(f"Failed to initialize environment configuration: {e}")


# Auto-initialize when module is imported
init_environment()