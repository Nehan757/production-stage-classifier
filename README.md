# Production Stage Classifier

Classifies film/TV production text into **development**, **pre_production**, or **production**, with a confidence score and short reasoning.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
python3 classifier.py "Principal photography began Monday at Pinewood Studios."
```

Output:
```
Stage:      production
Confidence: 100%
Reasoning:  The phrase 'Principal photography began' indicates that filming is currently in progress.
Method:     llm
```

## Architecture

```
Input text
    │
    ▼
Classifier agent (GPT-4o-mini)
    │
    ├── confidence ≥ 0.75 ──────────────────────────► Return result  (1 LLM call)
    │
    └── confidence < 0.75
            │
            ▼
        Critic agent (GPT-4o-mini)  ───────────────► Return verified result  (2 LLM calls)
```

**Why this approach?**

A rule-based fast-path was considered but discarded. Rules fail silently on negation ("no filming underway"), future tense ("will begin principal photography"), metaphor ("shooting for the stars"), and past-project references — all of which appear regularly in real production data. Patching each failure mode produces an unmaintainable system.

LLM-only with a Reflection step handles these naturally:

- **Classifier agent**: reads the text, outputs stage + confidence + one-line reasoning
- **Critic agent**: only fires when confidence < 0.75 — verifies tense, negation, and context before confirming or overriding the classification
- **Cost**: most clear-signal inputs never reach the critic (~1 LLM call each); only genuinely ambiguous text pays for 2 calls

## Stage Definitions

| Stage | When it applies |
|-------|----------------|
| `development` | Script deals, IP acquisition, pitching, greenlight decisions, financing, talent "attached", writers' room — no crew or cameras yet |
| `pre_production` | Casting calls, location scouting, crew assembly, production design, table reads, rehearsals, shoot scheduling — cameras have not rolled |
| `production` | Principal photography underway or completed — on-set activity, filming in progress, wrap announcements, dailies |

**Tense is critical:**
- "will begin principal photography" → `pre_production`
- "principal photography began" → `production`
- "the shoot wrapped last year" (past project) → look at what's happening *now*

## Usage as a library

```python
from classifier import classify

result = classify("Day 3 of shooting — second unit doing pickups downtown.")
print(result.stage)       # "production"
print(result.confidence)  # 1.0
print(result.reasoning)   # "The phrase 'day 3 of shooting' indicates..."
print(result.method)      # "llm"
```

Pass a pre-built client to avoid re-instantiation in bulk workloads:

```python
from openai import OpenAI
from classifier import classify

client = OpenAI()
for text in batch:
    result = classify(text, client=client)
```

## Run tests

```bash
python3 tests.py --offline   # unit tests only — no API key needed, fully deterministic
python3 tests.py             # unit tests + 28 integration tests (requires OPENAI_API_KEY)
```

The test suite has two layers:
- **Unit tests** (offline): mock `_call()` to verify pipeline logic — reflection triggering, API failure fallback, invalid stage coercion. Deterministic, no API cost.
- **Integration tests** (live): call the real API to validate prompt and model behaviour across 28 cases.

## Test coverage

28 cases across three difficulty tiers:

**Standard** — clear single-stage signals (greenlit, casting call, principal photography, wrap)

**Ambiguous** — competing signals, vague prose, minimal text

**Hard edge cases** — designed to break naive classifiers:

| Case | Challenge |
|------|-----------|
| Negation | "no filming underway" — keyword present but denied |
| Past project reference | "shoot wrapped two years ago" — historical, not current stage |
| Future principal photography | "will begin principal photography" — strong keyword, wrong tense |
| Metaphor | "shooting for the stars" — figurative use of a production keyword |
| `production` as a noun | "sources close to the production" — project noun, not stage |
| Mixed timeline | Two seasons, two stages in one sentence |
| Implicit on-set | No film keywords — inferred from crew, costumes, lighting rigs |
| Executive vague-speak | Corporate non-answer with no stage signal |
| Writers' room | Looks active but is development, not pre-production |
| Greenlight + shoot date | Straddles the dev → pre-production transition |

## Cost & reliability

- **Per call**: ~$0.0001 (GPT-4o-mini). A batch of 10,000 classifications costs ~$1–2 at real-time rates; use OpenAI's [Batch API](https://platform.openai.com/docs/guides/batch) for async workloads to cut that by 50%.
- **Reflection threshold** (`_REFLECTION_THRESHOLD = 0.75`): raise it to trigger the critic more often; lower it to skip reflection on borderline cases.
- **Retries**: `_call()` retries up to 3 times with exponential backoff on any API or parse error, then returns a `confidence=0.0` fallback — the pipeline never crashes on a bad response.
- **Uncertainty**: confidence below 0.5 signals genuinely ambiguous input — callers can flag these for human review rather than acting on them blindly.

## Files

```
classifier.py   — classifier + critic agents, public classify() function
models.py       — ClassificationResult dataclass
tests.py        — 28 test cases across standard, ambiguous, and hard edge cases
requirements.txt
.env.example
```
