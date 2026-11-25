"""A worked solution in the agent's 4-part format: Given -> Formulas -> Steps -> Interpretation."""
from __future__ import annotations

from dataclasses import dataclass, field


def fmt(x, d=2):
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e12):
        return f"{int(x):,}"
    return f"{x:,.{d}f}"


@dataclass
class Solution:
    title: str
    given: dict = field(default_factory=dict)
    formulas: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    interpretation: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)

    def step(self, text: str):
        self.steps.append(text)
        return self

    def render(self) -> str:
        L = [f"## {self.title}", "", "**1. Given**"]
        L += [f"- {k}: {v}" for k, v in self.given.items()]
        L += ["", "**2. Conditions / formulas**", *[f"- {f}" for f in self.formulas]]
        L += ["", "**3. Step-by-step**", *[f"{i}. {s}" for i, s in enumerate(self.steps, 1)]]
        if self.assumptions:
            L += ["", "**Assumptions**", *[f"- {a}" for a in self.assumptions]]
        L += ["", "**4. Interpretation**", *[f"- {s}" for s in self.interpretation],
              "", "_Analytical illustration, not financial advice._"]
        return "\n".join(L)
