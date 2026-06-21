# LLM Inference API

A minimal **FastAPI** service that serves a Hugging Face text-generation model (`distilgpt2`) over a REST API.

## What this demonstrates

- Setting up a Python environment for ML workloads
- Loading and running a Hugging Face transformer model
- Building a REST API (FastAPI) for AI inference
- Request validation, error handling, and structured responses with Pydantic

## Tech stack

- **FastAPI** — REST API framework
- **Hugging Face Transformers** — model loading and inference
- **PyTorch** — backend for model execution
- **Uvicorn** — ASGI server

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/kanika29082008-collab/llm-inference-api.git
cd llm-inference-api
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`.

> **Note:** the first request will take a little longer, since the `distilgpt2` model (~330MB) needs to download and load into memory.

## Usage

Open **http://localhost:8000/docs** in your browser for an interactive Swagger UI where you can try the API directly.

### Example request

**POST** `/generate`

```json
{
  "prompt": "The future of AI is",
  "max_new_tokens": 50,
  "temperature": 0.8
}
```

### Example response

```json
{
  "prompt": "The future of AI is",
  "generated_text": "The future of AI is bright, with new breakthroughs...",
  "model": "distilgpt2"
}
```

## Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|---------------------------------------|
| GET    | `/`         | Health check / welcome message        |
| GET    | `/health`   | Service and model status              |
| POST   | `/generate` | Generate text from a prompt           |

### Request parameters (`POST /generate`)

| Field            | Type   | Default | Description                                  |
|-------------------|--------|---------|-----------------------------------------------|
| `prompt`          | string | —       | Input text prompt (required)                  |
| `max_new_tokens`  | int    | 50      | Max tokens to generate (1–200)                |
| `temperature`     | float  | 0.8     | Sampling temperature (0.0–2.0)                |

## Project structure