"""
Production stage classifier — LLM-only with Reflection.

Two agents:
  1. Classifier  — reads the text, outputs stage + confidence + reasoning
  2. Critic      — only runs when confidence < 0.75; verifies or overrides

This keeps cost low (most clear cases never hit the critic) while handling
edge cases like negation, tense, and sarcasm that pure rules cannot.
"""

import json
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from models import ClassificationResult, Stage

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 2000
_REFLECTION_THRESHOLD = 0.75
_FALLBACK = {"stage": "development", "confidence": 0.0, "reasoning": "API error — could not classify."}

_CLASSIFIER_PROMPT = """You are an expert analyst of film and television production data.
Classify the text into exactly one stage:

- development: script deals, IP acquisition, pitching, greenlight, financing,
  talent "attached" before active prep. Writers' rooms are development, not pre-production.
  No crew, no locations locked, no cameras.
- pre_production: active prep before cameras roll — casting calls, location scouting,
  crew assembly, table reads, rehearsals, shoot scheduling, production designer hired.
- production: cameras rolling or already rolled — principal photography underway,
  on-set activity NOW, filming in progress, wrap announcements, dailies.

Tense is critical — apply these rules strictly:
- "will begin principal photography" / "set to start filming" = pre_production (not started)
- "principal photography began" / "cameras are rolling" = production (started)
- Past references to a finished project ("the shoot wrapped last year") describe history,
  not the current stage — look at what is happening NOW in the text.

Output JSON only:
{"stage": "development|pre_production|production", "confidence": 0.0-1.0, "reasoning": "one sentence citing specific phrases"}"""

# Critic uses a fixed system prompt; user-supplied text goes in the user turn
_CRITIC_SYSTEM = """You are a strict reviewer checking a production stage classification.
Does this classification correctly handle tense, negation, and context?
If correct, confirm it. If wrong, provide the right answer.

Output JSON only:
{"stage": "development|pre_production|production", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


def _call(client: OpenAI, system: str, user: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=150,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            # Validate required fields are present and well-typed
            stage = data["stage"]
            float(data["confidence"])
            str(data["reasoning"])
            return data
        except RateLimitError as exc:
            wait = 2 ** (attempt + 2)  # longer backoff for rate limits
            logger.warning("Rate limit (attempt %d/%d), retrying in %ds: %s", attempt + 1, retries, wait, exc)
            if attempt < retries - 1:
                time.sleep(wait)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON — not retrying: %s", exc)
            return _FALLBACK
        except Exception as exc:
            logger.warning("API call failed (attempt %d/%d): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return _FALLBACK


def classify(text: str, client: Optional[OpenAI] = None) -> ClassificationResult:
    """
    Classify production stage from messy text.

    Step 1 — Classifier agent: always runs.
    Step 2 — Critic agent: only runs when classifier confidence < 0.75.
    """
    if not text or not text.strip():
        return ClassificationResult(
            stage="development",
            confidence=0.0,
            reasoning="Empty input — cannot classify.",
            method="none",
        )

    if len(text) > _MAX_INPUT_CHARS:
        logger.warning("Input truncated from %d to %d chars.", len(text), _MAX_INPUT_CHARS)
        text = text[:_MAX_INPUT_CHARS]

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        client = OpenAI(api_key=api_key)

    try:
        # Step 1: classify
        data = _call(client, _CLASSIFIER_PROMPT, f"Classify:\n\n{text}")
        stage = data["stage"]
        confidence = round(float(data["confidence"]), 2)
        reasoning = data["reasoning"]
        method = "llm"

        # Step 2: reflect only when uncertain
        if confidence < _REFLECTION_THRESHOLD:
            logger.info("Low confidence (%.2f) on '%s...' — invoking critic.", confidence, text[:60])
            critic_user = (
                f"Text: {text}\n\n"
                f"Proposed: {stage} (confidence: {confidence})\n"
                f"Reasoning: {reasoning}\n\n"
                "Review this classification."
            )
            data = _call(client, _CRITIC_SYSTEM, critic_user)
            stage = data["stage"]
            confidence = round(float(data["confidence"]), 2)
            reasoning = data["reasoning"]
            method = "llm+reflection"

        if stage not in ("development", "pre_production", "production"):
            logger.warning("Invalid stage %r returned by LLM — coercing to fallback.", stage)
            stage = "development"
            confidence = 0.1

    except Exception as exc:
        logger.error("Unhandled error in classify(): %s", exc)
        return ClassificationResult(
            stage="development",
            confidence=0.0,
            reasoning=f"Unexpected error — could not classify: {exc}",
            method="llm",
        )

    return ClassificationResult(
        stage=stage,
        confidence=confidence,
        reasoning=reasoning,
        method=method,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)

    if len(sys.argv) < 2:
        print('Usage: python3 classifier.py "<text>"')
        sys.exit(1)

    print(classify(" ".join(sys.argv[1:])))
