from dataclasses import dataclass
from typing import Literal

Stage = Literal["development", "pre_production", "production"]


@dataclass
class ClassificationResult:
    stage: Stage
    confidence: float  # 0.0 – 1.0
    reasoning: str
    method: Literal["none", "llm", "llm+reflection"]  # which path was taken

    def __str__(self) -> str:
        return (
            f"Stage:      {self.stage}\n"
            f"Confidence: {self.confidence:.0%}\n"
            f"Reasoning:  {self.reasoning}\n"
            f"Method:     {self.method}"
        )
