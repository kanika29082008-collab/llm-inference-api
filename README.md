# LLM Inference API

A minimal FastAPI service that serves a Hugging Face text-generation model (distilgpt2) over a REST API.

## What this demonstrates
- Setting up a Python environment for ML workloads
- Loading and running a Hugging Face transformer model
- Building an API (FastAPI) for AI inference
- Request validation, error handling, and structured responses

## Tech stack
- FastAPI - REST API framework
- Hugging Face Transformers - model loading and inference
- PyTorch - backend for model execution
- Uvicorn - ASGI server

## Setup

pip install -r requirements.txt
uvicorn main:app --reload

The server starts at http://localhost:8000

## Usage

Open http://localhost:8000/docs in your browser for an interactive test UI.

### Example request
POST /generate
{
  "prompt": "The future of AI is",
  "max_new_tokens": 50,
  "temperature": 0.8
}

## Endpoints
- GET / - health check
- GET /health - service and model status
- POST /generate - generate text from a prompt
