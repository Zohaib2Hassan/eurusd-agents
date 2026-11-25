"""Automated step grading (deterministic) and the interactive practice loop.

Rubric per step (weight = 100 / steps):
  numeric target: |error| <= tol -> 100%;  <= 2% -> 85% (minor slip);  <= 10% -> 65% (minor slip);
                  a number given but further off -> 35% (conceptual/algebraic flaw);  no number -> 0%
  step score = mean over targets;   text steps: coverage of concept groups (>= min_concepts -> 100%)
  hint used on the step: max 80% of the step;  "show solution": 0 for this and all remaining steps
  pass mark per step 75%; mastery tiers 90/75/50.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bank import BANK, Problem, Step

NUM = re.compile(r"[-+]?\$?\s?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?\s?(?:k|K)?")


def extract_numbers(text: str) -> list[float]:
    out = []
    text = re.sub(r"\b[A-Za-z_]+\d+\*?", " ", text.replace("−", "-"))   # drop labels like Q1*, R2, x2
    text = re.sub(r"[²³]", " ", text)
    for m in NUM.findall(text):
        t = m.replace("$", "").replace(",", "").replace(" ", "")
        mult = 1000 if t.lower().endswith("k") else 1
        t = t.rstrip("kK")
        try:
            out.append(float(t) * mult)
        except ValueError:
            pass
    return out


def _target_score(nums: list[float], target) -> tuple[float, float | None]:
    if not nums:
        return 0.0, None
    cands = list(nums)
    if target.kind == "percent":
        cands += [n * 100 for n in nums]          # accept 0.095 for 9.5%
    if target.value < 0:
        cands += [-abs(n) for n in nums]
    best = min(cands, key=lambda n: abs(n - target.value) / max(abs(target.value), 1e-9))
    err = abs(best - target.value) / max(abs(target.value), 1e-9)
    if err <= target.tol:
        return 1.0, best
    if err <= 0.02:
        return 0.85, best
    if err <= 0.10:
        return 0.65, best
    return 0.35, best


def _diagnose(step: Step, nums: list[float], scores: list[float]) -> list[str]:
    """Match the student's numbers against known wrong answers (only when not already fully correct)."""
    if all(sc >= 1.0 for sc in scores):
        return []
    msgs = []
    for wrong, msg in step.misconceptions:
        if any(abs(n - wrong) <= 0.005 * max(abs(wrong), 1e-9) for n in nums) and \
                not any(abs(wrong - t.value) <= t.tol * max(abs(t.value), 1e-9) for t in step.targets):
            msgs.append(msg)
    return msgs


def grade_step(step: Step, answer: str, hint_used: bool = False) -> dict:
    detail = []
    if step.targets:
        nums = extract_numbers(answer)
        scores = []
        remaining = list(nums)
        for t in step.targets:
            sc, used = _target_score(remaining or nums, t)
            if used is not None and used in remaining:
                remaining.remove(used)
            scores.append(sc)
            detail.append({"label": t.label, "expected": round(t.value, 6), "matched": used, "score": sc})
        frac = sum(scores) / len(scores)
        diag = _diagnose(step, nums, scores)
        if diag:
            detail.append({"diagnosis": diag})
            frac = min(frac, 0.59)          # a recognised conceptual error is never a "minor slip"
    else:
        a = answer.lower()
        hits = [any(k in a for k in grp) for grp in step.concepts]
        need = step.min_concepts or len(step.concepts)
        frac = min(1.0, sum(hits) / need)
        detail = [{"concept": grp[0], "found": h} for grp, h in zip(step.concepts, hits)]
    raw = frac
    band = ("Full credit" if raw >= 0.9 else "Partial - minor slip" if raw >= 0.6 else
            "Partial - conceptual/algebraic flaw" if raw >= 0.25 else "No credit")
    if hint_used and frac > 0.8:
        frac = 0.8
        band += " (hint used: capped at 80%)"
    return {"fraction": round(frac, 4), "band": band, "detail": detail, "passed": frac >= 0.75}


def tier(pct: float) -> str:
    return "Mastery" if pct >= 90 else "Proficient" if pct >= 75 else "Developing" if pct >= 50 else "Needs Review"


@dataclass
class Session:
    problem: Problem
    step_i: int = 0
    hints: set = field(default_factory=set)
    scores: list = field(default_factory=list)   # best fraction per completed step
    attempts: dict = field(default_factory=dict)
    revealed: bool = False
    quit: bool = False

    @property
    def weight(self) -> float:
        return 100 / len(self.problem.steps)

    def done(self) -> bool:
        return self.step_i >= len(self.problem.steps)

    def submit(self, answer: str) -> dict:
        st = self.problem.steps[self.step_i]
        g = grade_step(st, answer, self.step_i in self.hints)
        self.attempts[self.step_i] = self.attempts.get(self.step_i, 0) + 1
        g["points"] = round(g["fraction"] * self.weight, 2)
        g["weight"] = self.weight
        if g["passed"] or self.attempts[self.step_i] >= 3:     # move on after a pass or 3 tries
            self.scores.append(g["fraction"])
            g["advance"] = True
            g["solution"] = st.solution if not g["passed"] else None
            self.step_i += 1
        else:
            g["advance"] = False
        return g

    def hint(self) -> str:
        self.hints.add(self.step_i)
        return self.problem.steps[self.step_i].hint

    def show_solution(self) -> list[str]:
        self.revealed = True
        out = []
        while not self.done():
            st = self.problem.steps[self.step_i]
            out.append(f"Step {self.step_i + 1}: {st.solution}" + (f"\n   NOTE: {st.correction}" if st.correction else ""))
            self.scores.append(0.0)
            self.step_i += 1
        return out

    def report(self) -> str:
        w = self.weight
        rows = [f"| Step {i + 1} | {self.problem.steps[i].prompt[:60]} | {s * w:.1f} / {w:.0f} |" for i, s in enumerate(self.scores)]
        total = sum(self.scores) * w
        return "\n".join(["| Step | Task | Score |", "|---|---|---|", *rows, f"| **Total** | | **{total:.1f} / 100** |", "",
                          f"**Final score: {total:.1f}% - {tier(total)}**", "", f"Takeaway: {self.problem.takeaway}"])


def run_interactive(pid=None, inp=input, out=print, problem: Problem | None = None, record_progress: bool = True):
    import random
    p = problem or (BANK[pid] if pid else BANK[random.choice(list(BANK))])
    s = Session(p)
    tag = f"seed {p.seed}" if p.seed is not None else f"bank #{p.pid}"
    out(f"\n### {p.topic} ({tag})  |  Difficulty: {p.difficulty}  |  {len(p.steps)} steps × {s.weight:.0f} pts")
    out(f"Given: {p.given}\nCommands: 'hint' (caps the step at 80%), 'show solution' (0 for remaining steps), 'quit'\n")
    while not s.done():
        st = p.steps[s.step_i]
        out(f"Step {s.step_i + 1}: {st.prompt}")
        try:
            a = inp("> ").strip()
        except (EOFError, KeyboardInterrupt):
            a = "quit"
        if a.lower() in ("quit", "exit"):
            out("Session ended."); s.quit = True; return s
        if a.lower() == "hint":
            out(f"Hint: {s.hint()}"); continue
        if a.lower() == "show solution":
            out("\n".join(s.show_solution())); break
        g = s.submit(a)
        out(f"**Step {s.step_i if g['advance'] else s.step_i + 1} Score: {g['points']:.1f} / {g['weight']:.0f} pts** ({g['band']})")
        if g["advance"]:
            if g.get("solution"):
                out(f"Worked answer: {g['solution']}")
            if st.correction:
                out(f"Note on the answer key: {st.correction}")
            out(f"Running total: {sum(s.scores) * s.weight:.1f} / {len(s.scores) * s.weight:.0f}\n")
        else:
            wrong = [d for d in g["detail"] if ("label" in d and d["score"] < 0.75) or d.get("found") is False]
            diags = [m for d in g["detail"] for m in d.get("diagnosis", [])]
            out(f"Not yet - check {', '.join(d.get('label') or d.get('concept') for d in wrong) or 'your working'}.")
            for m in diags:
                out(f"Likely mistake: {m}")
            if not diags:
                out(f"Hint: {st.hint}")
            out("Try this step again.\n")
    out("\n" + s.report())
    if record_progress:
        from .progress import record
        record(p.topic_key, p.topic, p.seed, sum(s.scores) * s.weight, len(s.hints), s.revealed)
    return s
