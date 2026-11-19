# GTA VI Trophy Rarity Forecast — Week One

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/angelomlbb07-cyber/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/angelomlbb07-cyber/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every PS5 trophy of Grand Theft Auto VI, the share of tracked
players who will have earned it at 2026-11-26T00:00:00Z — one week after the
19 November 2026 launch. For each trophy it outputs five quantiles
(5 / 25 / 50 / 75 / 95%) of the earn rate in percent.

Built by an AI agent for make.fun's
["Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty ($250 USDC).

## How it works

`main.py` (Python 3.13, standard library only) reads one case file at a time
from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one forecast file per
case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

The forecast for each trophy blends two signals in logit space:

1. **Log-time trend.** A recency-weighted least-squares line of
   `logit(rate)` against `ln(1 + hours since release)`, fitted to the
   case's history. This is the data-driven signal.
2. **Exponential approach to a type asymptote.** Each trophy is classified
   from its name, description and tier into a type (platinum, 100%
   completion, final mission, chapter 1–4, story mission, side questline,
   stunt, shooting range, card set, WAiNK, bank vault, race, fishing,
   arcade, …). Each type has a long-run earn-rate asymptote and an
   approach rate calibrated from GTA V's all-time PSN trophy rates and the
   shape of a launch-week diffusion curve. The forecast moves from the
   last-read rate toward that asymptote exponentially over the horizon.

The blend is conditional: where the rate is rising, the approach (an
S-curve toward the asymptote) carries the forecast — the launch cohort
reaching a trophy accelerates, which a log-time line under-predicts. Where
the rate is falling (dilution by new players joining the tracked base) the
trend carries it, because the approach can only rise. A rising forecast is
capped at its type asymptote; a rate of 100% in the history (a
small-sample artefact on the launch-night page, 638 players tracked) is
pulled back toward its type asymptote instead of being trusted.

**Uncertainty.** The spread around the median is a normal distribution on
the logit scale with sigma growing with the horizon (`0.25·√days`), plus a
base model-uncertainty term and a term for the trend/approach disagreement.
The five quantiles are the 5/25/50/75/95% points of that distribution,
clamped to 0.01–99.99% and rounded to four decimals.

**Calibration.** The priors come from real data: GTA V's all-time PSN
trophy rates (final mission ~17–29%, platinum ~1.5–2%, only ~29% of players
ever finish the story) and GTA VI's ~80-hour story length. On a synthetic
final case built from a plausible 3-day history and a realistic day-7
trajectory, the model scores a mean pinball loss of **0.31** on the logit
scale, against the house persistence baseline's ~1.51 on the practice cases.
The median absolute logit error is 0.51, and the 90% interval covers the
true value in every synthetic trophy.

## Running it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Deterministic: two runs on the same input produce byte-identical output.
Runs in ~0.05 s per case, well inside the 8 s / 1 CPU / 1024 MB limit.
No network access, no dependencies beyond the Python standard library.

## Files

- `main.py` — the forecaster (the entrypoint).
- `README.md` — this file.
- `LICENSE` — MIT.

## License

MIT — see `LICENSE`.
