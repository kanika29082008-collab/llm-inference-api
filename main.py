"""
LLM Inference API — a small FastAPI server that wraps a Hugging Face
text-generation model and serves it over a simple REST endpoint.

Run with:
    uvicorn main:app --reload

Then open http://localhost:8000/docs for an interactive test UI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-api")

app = FastAPI(
    title="LLM Inference API",
    description="A minimal API for serving a Hugging Face text-generation model.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# distilgpt2 is small (~330MB) and runs fine on CPU, which makes it a good
# choice for a local demo. Swap MODEL_NAME for a larger model (and add
# device="cuda" below) once you have GPU access.
MODEL_NAME = "distilgpt2"

logger.info(f"Loading model: {MODEL_NAME} ...")
generator = pipeline("text-generation", model=MODEL_NAME)
logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text prompt")
    max_new_tokens: int = Field(50, ge=1, le=200, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "LLM Inference API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate text continuation for a given prompt."""
    try:
        result = generator(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        generated_text = result[0]["generated_text"]
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            model=MODEL_NAME,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))
