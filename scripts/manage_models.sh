#!/bin/bash
# Model management script for Maternal Instinct AI

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "Ollama Model Management for Maternal Instinct AI"
echo "================================================"
echo "Ollama Host: $OLLAMA_HOST"
echo ""

# Check if Ollama is running
if ! curl -s "$OLLAMA_HOST/api/version" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to Ollama at $OLLAMA_HOST"
    echo "Please make sure Ollama is running."
    exit 1
fi

case "$1" in
    list)
        echo "Available models:"
        curl -s "$OLLAMA_HOST/api/tags" | python -m json.tool
        ;;
    pull)
        if [ -z "$2" ]; then
            echo "Usage: $0 pull <model_name>"
            echo "Example: $0 pull llama3.1:8b"
            exit 1
        fi
        echo "Pulling model: $2"
        curl -X POST "$OLLAMA_HOST/api/pull" -d "{\"name\":\"$2\"}"
        ;;
    remove)
        if [ -z "$2" ]; then
            echo "Usage: $0 remove <model_name>"
            exit 1
        fi
        echo "Removing model: $2"
        curl -X DELETE "$OLLAMA_HOST/api/delete" -d "{\"name\":\"$2\"}"
        ;;
    pull-required)
        echo "Pulling required models for Maternal Instinct AI..."
        echo ""
        echo "Pulling llama3.1:8b (LLM model)..."
        curl -X POST "$OLLAMA_HOST/api/pull" -d '{"name":"llama3.1:8b"}'
        echo ""
        echo "Pulling bge-m3 (Embedding model)..."
        curl -X POST "$OLLAMA_HOST/api/pull" -d '{"name":"bge-m3"}'
        echo ""
        echo "Done! Models installed:"
        curl -s "$OLLAMA_HOST/api/tags" | python -m json.tool
        ;;
    *)
        echo "Usage: $0 {list|pull|remove|pull-required}"
        echo ""
        echo "Commands:"
        echo "  list           - List all installed models"
        echo "  pull <model>   - Pull a specific model"
        echo "  remove <model> - Remove a specific model"
        echo "  pull-required   - Pull all models required by Maternal Instinct AI"
        exit 1
        ;;
esac