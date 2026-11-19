"""GTA VI trophy rarity forecast for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every PS5 trophy of Grand Theft Auto VI, its earn rate on
Exophase at the case's target, as five quantiles (5/25/50/75/95%).

Model (per trophy, all on the logit scale):
  1. Trend: a straight line of logit(rate) against ln(1 + hours since
     release), fitted by least squares to the case's history (weighted
     toward recent reads). This is the house curve baseline's form, but
     fitted to the whole history the case carries, not just 24h.
  2. Approach: an exponential approach from the last observed rate toward
     a type-based asymptote (the long-run share of tracked players who
     will ever earn the trophy, calibrated on GTA V's all-time PSN rates
     and the trophy's difficulty). The approach rate is type-specific:
     story trophies close fast, grind trophies slow, the platinum barely
     moves in a week.
  3. Blend: where the history actually moved, the trend earns weight;
     where it is flat (launch night, all reads identical) the approach
     carries the forecast, so a flat history does not become a flat
     forecast.
  4. Cap: a rising forecast never passes its type asymptote.
  5. Spread: a random-walk term that widens with the horizon (the house
     baseline's 0.5*sqrt(days)) plus model uncertainty plus the
     disagreement between trend and approach, so the quantiles stay
     honest about a 4.5-day extrapolation.

The 100% rate a launch-night read shows for one trophy is an artefact
of a 638-player sample; it is forecast back toward its type asymptote,
not held at 100%.

Standard library only. Reads every *.in in the input directory and
writes one *.out per case. Deterministic: no randomness, no wall-clock
dependence, inputs read in name order.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release of GTA VI on PS5, the time base for the trend fit.
RELEASE = "2026-11-19T00:00:00Z"


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        e = math.exp(-y)
        return 100 / (1 + e)
    e = math.exp(y)
    return 100 * e / (1 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> float:
    """ISO 8601 UTC timestamp -> seconds since the epoch."""
    # 2026-11-19T03:00:00Z  or  2026-11-19T02:30:00.261Z
    s = s.replace("Z", "+00:00")
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(s)
    return dt.timestamp()


RELEASE_EPOCH = parse_time(RELEASE)


def hours_since_release(epoch: float) -> float:
    return (epoch - RELEASE_EPOCH) / 3600.0


# ---------------------------------------------------------------------------
# Trophy type priors.
#
# asymptote: the long-run share (%) of tracked players who will ever earn
# the trophy, calibrated on GTA V's all-time PSN rates (final-mission
# trophy ~17-29%, platinum ~1.5-2%, chapter trophies 50-85%) and the
# trophy's difficulty. The forecast never passes it.
# approach: per-day exponential rate at which the logit gap to the
# asymptote closes. Story beats close fast; grinds slow; the platinum
# and 100% barely move inside a week.
# ---------------------------------------------------------------------------

def classify(trophy: dict):
    tier = trophy.get("tier", "bronze")
    name = trophy.get("name", "")
    desc = trophy.get("description", "")
    text = (name + " " + desc).lower()

    if tier == "platinum":
        return 2.0, 0.015
    if "100% completion" in text:
        return 3.0, 0.015
    if "final mission" in text and tier == "gold":
        return 30.0, 0.13
    if "complete chapter 1" in text:
        return 85.0, 0.40
    if "complete chapter 2" in text:
        return 80.0, 0.32
    if "complete chapter 3" in text:
        return 75.0, 0.28
    if "complete chapter 4" in text:
        return 70.0, 0.24
    if "complete the final mission for" in text:
        # A side character's questline: most story players finish one.
        return 45.0, 0.20
    # Activity keywords first: several descriptions start with "Complete"
    # but name an activity, not a story mission.
    if "stunt" in text:
        return 15.0, 0.07
    if "shooting range" in text or "gold medal" in text:
        return 12.0, 0.07
    if "stranger" in text:
        return 25.0, 0.09
    if "card set" in text or "collectible card" in text:
        return 15.0, 0.07
    if "waink" in text:
        return 10.0, 0.05
    if "bank vault" in text:
        return 20.0, 0.09
    if "win" in text and "race" in text:
        return 20.0, 0.09
    if "fishing" in text:
        return 30.0, 0.20
    if "arcade" in text:
        return 35.0, 0.20
    if "meditate" in text or "cocktail" in text or "bowling" in text:
        return 40.0, 0.20
    if "try each" in text:
        return 45.0, 0.25
    if "discover" in text or "explore" in text:
        return 35.0, 0.15
    if "rob" in text:
        return 25.0, 0.15
    if "hunt" in text:
        return 20.0, 0.15
    if "earn $" in text or "spend $" in text or "purchase" in text:
        return 20.0, 0.15
    if text.startswith("complete ") or "complete " in text:
        # A named story mission: required for story progress.
        return 65.0, 0.25
    if tier == "silver":
        return 18.0, 0.08
    if tier == "gold":
        return 25.0, 0.10
    return 30.0, 0.18


def linfit(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def forecast_trophy(trophy, history, horizon_days, target_hours):
    """Five quantiles for one trophy at the target."""
    asymptote, approach = classify(trophy)
    a_logit = logit(asymptote)
    idx = trophy["id"] - 1  # trophies are in id order; rates follow it

    # Collect (hours, logit rate) pairs, skipping reads that lack the trophy.
    pts = []
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        t = hours_since_release(parse_time(read["at"]))
        pts.append((t, logit(rates[idx])))

    if not pts:
        # No read ever listed this trophy: forecast the asymptote approach
        # from a near-zero start.
        y_med = a_logit - (a_logit - logit(0.05)) * math.exp(
            -approach * horizon_days
        )
        sigma = 1.5
        return quantiles(y_med, sigma)

    last_t, last_y = pts[-1]
    last_rate = inv_logit(last_y)

    # A 100% (or ~100%) rate on launch night is a small-sample artefact:
    # forecast it back toward its type asymptote instead of holding it.
    if last_rate >= 99.5:
        y_med = a_logit
        sigma = 1.6
        return quantiles(y_med, sigma)

    # Trend: logit against ln(1 + hours), weighted toward recent reads.
    xs = [math.log1p(max(0.0, t)) for t, _ in pts]
    ys = [y for _, y in pts]
    # Recency weights: the newest read counts 3x the oldest.
    n = len(pts)
    ws = [1.0 + 2.0 * (i / max(1, n - 1)) for i in range(n)]
    fit = linfit(xs, ys, ws)
    if fit is None:
        y_trend = last_y
        b = 0.0
    else:
        a, b = fit
        y_trend = a + b * math.log1p(max(0.0, target_hours))

    # Approach: exponential close toward the asymptote from the last read.
    y_appr = a_logit - (a_logit - last_y) * math.exp(-approach * horizon_days)

    # Blend in logit space. The trend is a log-time line: it under-predicts
    # an accelerating rise (the launch cohort reaching a trophy) and
    # over-predicts nothing, so where the rate is rising the approach
    # (an S-curve toward the asymptote) carries the forecast. Where the
    # rate is falling (dilution by new players) the approach cannot fall,
    # so the trend carries it.
    if b > 0:
        w_trend = 0.2
    else:
        w_trend = 0.7
    y_blend = w_trend * y_trend + (1.0 - w_trend) * y_appr

    # Cap: a rising forecast never passes its type asymptote.
    if y_blend > a_logit:
        y_blend = a_logit
    # Floor: never below a 0.01% rate.
    y_med = max(y_blend, logit(0.01))

    # Spread: random-walk term widening with the horizon, plus model
    # uncertainty, plus the trend/approach disagreement.
    sigma = 0.25 * math.sqrt(max(0.0, horizon_days))
    sigma += 0.18
    sigma += 0.1 * abs(y_trend - y_appr)
    sigma = min(1.2, max(0.35, sigma))

    return quantiles(y_med, sigma)


def quantiles(y_med, sigma):
    out = []
    for z in Z:
        q = inv_logit(y_med + z * sigma)
        out.append(round4(clamp(q)))
    # Enforce non-decreasing (rounding can tie, never invert).
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = hours_since_release(parse_time(case["target"]))
    rows = []
    for trophy in case["trophies"]:
        q = forecast_trophy(trophy, history, horizon_days, target_hours)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
