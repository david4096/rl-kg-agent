#!/usr/bin/env python3
"""Test SSL configuration and model downloads."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ssl_config():
    """Test SSL configuration."""
    print("🔐 Testing SSL Configuration...")
    
    ssl_cert = os.getenv("SSL_CERT_FILE")
    if ssl_cert:
        print(f"✅ SSL_CERT_FILE: {ssl_cert}")
        if os.path.exists(os.path.expanduser(ssl_cert)):
            print("✅ SSL certificate file exists")
        else:
            print("❌ SSL certificate file not found")
    else:
        print("❌ SSL_CERT_FILE not set")
    
    requests_bundle = os.getenv("REQUESTS_CA_BUNDLE")
    if requests_bundle:
        print(f"✅ REQUESTS_CA_BUNDLE: {requests_bundle}")
    
    return ssl_cert is not None


def test_simple_download():
    """Test a simple HTTPS download."""
    print("\n🌐 Testing HTTPS Download...")
    
    try:
        import requests
        response = requests.get("https://httpbin.org/status/200", timeout=10)
        if response.status_code == 200:
            print("✅ HTTPS request successful")
            return True
        else:
            print(f"❌ HTTPS request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ HTTPS request failed: {e}")
        return False


def test_model_download():
    """Test downloading a small model."""
    print("\n🤖 Testing Model Download...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a very small model for testing
        model_name = "all-MiniLM-L6-v2"
        print(f"Downloading {model_name}...")
        
        model = SentenceTransformer(model_name)
        print("✅ Model download successful")
        
        # Test encoding
        test_sentences = ["Hello world", "This is a test"]
        embeddings = model.encode(test_sentences)
        print(f"✅ Model encoding works: {len(embeddings)} embeddings, shape: {embeddings[0].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 SSL and Model Download Test\n")
    
    # Import environment config
    try:
        from rl_kg_agent.utils.env_config import load_environment_config
        load_environment_config()
        print("✅ Environment configuration loaded")
    except Exception as e:
        print(f"❌ Environment configuration failed: {e}")
    
    ssl_ok = test_ssl_config()
    download_ok = test_simple_download()
    
    if ssl_ok and download_ok:
        model_ok = test_model_download()
        
        if model_ok:
            print("\n🎉 All tests passed! SSL and model downloads are working.")
        else:
            print("\n⚠️ SSL works but model download failed.")
    else:
        print("\n❌ SSL configuration issues detected.")