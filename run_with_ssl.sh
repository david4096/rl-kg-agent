#!/bin/bash

# Export SSL certificates for Azure OpenAI connections
export SSL_CERT_FILE=/Users/markstreer/certs/combined-certs.pem
export REQUESTS_CA_BUNDLE=/Users/markstreer/certs/combined-certs.pem
export CURL_CA_BUNDLE=/Users/markstreer/certs/combined-certs.pem

# Disable HuggingFace tokenizer parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

echo "✅ SSL certificates configured:"
echo "   SSL_CERT_FILE=$SSL_CERT_FILE"
echo "   REQUESTS_CA_BUNDLE=$REQUESTS_CA_BUNDLE" 
echo "   CURL_CA_BUNDLE=$CURL_CA_BUNDLE"

# Run the RL-KG-Agent with proper environment
exec "$@"