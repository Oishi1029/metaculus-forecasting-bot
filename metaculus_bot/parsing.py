"""Extract a forecast from LLM prose. Deterministic first, LLM salvage last.

WHY THIS FILE EXISTS AND WHY IT IS PARANOID
Scoring sums peer scores and then squares the sum, so a question we fail to
forecast is a zero inside that sum -- strictly worse than a mediocre forecast.
A parse failure is therefore a scoring event, not a logging event. Hence the
ladder: strict JSON block -> repaired JSON -> loose regex -> LLM salvage.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

log = logging.getLogger(__name__)

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_OBJ = re.compile(r"\{[^{}]*\"question_type\"[^{}]*\}", re.DOTALL)


class ParseError(ValueError):
    pass


# --- generic block extraction ------------------------------------------------
def extract_json_block(text: str) -> dict[str, Any]:
    """Find the forecast JSON. Prompts require it LAST, so we search backwards."""
    if not text:
        raise ParseError("empty model output")

    candidates: list[str] = [m.group(1) for m in _FENCED.finditer(text)]
    candidates += [m.group(0) for m in _BARE_OBJ.finditer(text)]
    # Last resort: the final balanced {...} span in the whole response.
    tail = _last_balanced_object(text)
    if tail:
        candidates.append(tail)

    for raw in reversed(candidates):
        for attempt in (raw, _repair_json(raw)):
            try:
                obj = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
    raise ParseError("no parseable JSON object in model output")


def _last_balanced_object(text: str) -> str | None:
    depth = 0
    end = -1
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == "}":
            if depth == 0:
                end = i
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0 and end != -1:
                return text[i:end + 1]
    return None


def _repair_json(raw: str) -> str:
    """Fix the failure modes LLMs actually produce, in order of frequency."""
    s = raw.strip()
    s = re.sub(r"//[^\n]*", "", s)                    # line comments
    s = re.sub(r",\s*([}\]])", r"\1", s)              # trailing commas
    s = s.replace("'", '"')                           # single quotes
    s = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "null", s)
    s = re.sub(r"\b(True|False|None)\b",
               lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)], s)
    s = re.sub(r"(\d)\s*%", r"\1", s)                 # "35%" -> 35
    s = re.sub(r"(-?\d+),(\d{3})\b", r"\1\2", s)      # thousands separators
    return s


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v.replace(",", ""))
        if m:
            try:
                f = float(m.group(0))
                return f if math.isfinite(f) else None
            except ValueError:
                return None
    return None


# --- per-type extraction -----------------------------------------------------
def parse_binary(text: str) -> float:
    """Return P(yes) in (0, 1)."""
    try:
        obj = extract_json_block(text)
        p = _as_float(obj.get("posterior_prob"))
        if p is None:
            for k in ("probability", "prob", "p_yes", "probability_yes", "final_probability"):
                p = _as_float(obj.get(k))
                if p is not None:
                    break
    except ParseError:
        p = None

    if p is None:
        p = _loose_binary(text)
    if p is None:
        raise ParseError("no binary probability found")
    if p > 1.0:                       # model emitted a percentage
        p = p / 100.0
    if not (0.0 <= p <= 1.0):
        raise ParseError(f"binary probability out of range: {p}")
    return p


def _loose_binary(text: str) -> float | None:
    pats = [
        r"probability[^0-9%]{0,24}?(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:probability|chance|likely)",
        r"posterior_prob\D{0,12}(\d*\.?\d+)",
        r"final\s+(?:answer|forecast|probability)\D{0,16}(\d*\.?\d+)",
    ]
    for pat in pats:
        m = list(re.finditer(pat, text, re.IGNORECASE))
        if m:
            v = _as_float(m[-1].group(1))
            if v is not None:
                return v / 100.0 if v > 1.0 else v
    return None


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """Return {option: probability}, keys exactly matching ``options``.

    The Metaculus server matches option keys EXACTLY and the SDK performs no
    validation, so a mis-cased or reworded key fails server-side with no client
    warning. We map back onto the canonical strings here, not later.
    """
    obj = extract_json_block(text)
    raw = obj.get("option_probabilities") or obj.get("probabilities") \
        or obj.get("probability_yes_per_category") or obj.get("options")
    probs: dict[str, float] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            f = _as_float(v)
            if f is not None:
                probs[str(k)] = f
    elif isinstance(raw, list):
        # Either [0.2, 0.3, ...] or [{"option": "...", "probability": ...}, ...]
        if raw and isinstance(raw[0], dict):
            for item in raw:
                name = item.get("option") or item.get("name") or item.get("label")
                f = _as_float(item.get("probability", item.get("prob")))
                if name is not None and f is not None:
                    probs[str(name)] = f
        else:
            for i, v in enumerate(raw):
                if i < len(options):
                    f = _as_float(v)
                    if f is not None:
                        probs[options[i]] = f

    if not probs:
        raise ParseError("no multiple-choice probabilities found")

    # Canonicalise keys back onto the exact option strings.
    canon: dict[str, float] = {}
    lowered = {o.strip().lower(): o for o in options}
    for k, v in probs.items():
        target = lowered.get(str(k).strip().lower())
        if target is None:
            # tolerate minor rewording: unique substring match
            hits = [o for o in options if str(k).strip().lower() in o.lower()
                    or o.lower() in str(k).strip().lower()]
            target = hits[0] if len(hits) == 1 else None
        if target is not None:
            canon[target] = canon.get(target, 0.0) + v

    missing = [o for o in options if o not in canon]
    if len(missing) == len(options):
        raise ParseError("none of the returned option names matched the question")
    for o in missing:
        canon[o] = 0.0
        log.warning("multiple choice: model omitted option %r, assigned 0 before renormalising", o)
    return canon


def parse_numeric_percentiles(text: str, wanted: list[float]) -> dict[float, float]:
    """Return {percentile_0_100: value}. Accepts 0-1 or 0-100 percentile keys."""
    obj = extract_json_block(text)
    raw = obj.get("declared_percentiles") or obj.get("percentiles") or obj.get("quantiles")
    if not isinstance(raw, dict) or not raw:
        raise ParseError("no declared_percentiles object found")

    out: dict[float, float] = {}
    for k, v in raw.items():
        pk = _as_float(k)
        pv = _as_float(v)
        if pk is None or pv is None:
            continue
        if 0.0 < pk <= 1.0:          # 0.05 style
            pk *= 100.0
        if not (0.0 < pk < 100.0):
            continue
        out[round(pk, 4)] = pv

    if len(out) < 3:
        raise ParseError(f"only {len(out)} usable percentiles parsed (need >= 3)")
    return out


# --- salvage -----------------------------------------------------------------
SALVAGE_INSTRUCTION = (
    "The text below is a forecaster's report whose machine-readable block is "
    "missing or malformed. Read it and output ONLY the corrected JSON object, "
    "with no commentary and no code fence.\n"
    "Required shape for a {qtype} question:\n{shape}\n\n--- REPORT ---\n{report}"
)

SHAPES = {
    "binary": '{"question_type": "binary", "posterior_prob": <decimal 0-1>}',
    "multiple_choice": '{"question_type": "multiple_choice", "option_probabilities": '
                       '{"<exact option name>": <decimal 0-1>, ...}}',
    "numeric": '{"question_type": "numeric", "declared_percentiles": '
               '{"0.01": <value>, "0.025": <value>, "0.05": <value>, "0.1": <value>, '
               '"0.2": <value>, "0.4": <value>, "0.5": <value>, "0.6": <value>, '
               '"0.8": <value>, "0.9": <value>, "0.95": <value>, "0.975": <value>, '
               '"0.99": <value>}}',
}


def salvage_prompt(qtype: str, report: str) -> str:
    key = "numeric" if qtype in ("numeric", "discrete") else qtype
    return SALVAGE_INSTRUCTION.format(
        qtype=key, shape=SHAPES.get(key, SHAPES["binary"]), report=report[-8000:]
    )
