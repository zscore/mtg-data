# mtg-data

Shared 17Lands data layer for the MTG draft analyses. It downloads 17Lands public draft, game and replay
files, converts them to parquet, loads them as per-draft tensors, and holds the encodings that more than
one model uses. Data: [17Lands public datasets](https://www.17lands.com/public_datasets). Please attribute
17Lands and check their [usage guidelines](https://www.17lands.com/usage_guidelines).

Used by:

| repo | what it is |
|---|---|
| [wheelnet](../wheelnet) | wheel prediction and open-lane analysis (WheelNet) |
| [picknet](../picknet) | PickNet, the pick-prediction transformer |
| [mtg-deck-strength](../mtg-deck-strength) | deck strength, DraftEncoder replication, card synergy |
| [mtg-bombs](../mtg-bombs) | bombs analysis and blog post |
| [mtg-draft-sim-site](../mtg-draft-sim-site) | browser draft simulator (reads parquet and pick shards from here) |

## Install

In each analysis repo's venv:

```
pip install -e ../mtg-data
```

## Where the data lives

Everything is under `data/` in this repo (gitignored, about 26 GB for all sets). Set `MTG_DATA_DIR` to
use another location. `mtgdata.data` exposes the paths:

```
DATA       data/                 MTG_DATA_DIR or this repo's data/
RAW        data/raw/             17Lands .csv.gz downloads and Scryfall cards
PARQUET    data/parquet/         draft files as parquet, slim game/replay parquet
FEATURES   data/features/        per-set caches the analysis repos write (wheel features, text
                                 embeddings, DraftEncoder tokens) and PickNet's all-sets pick shards
```

Analysis repos write their own outputs to their own `results/`. Only reusable caches go in `FEATURES`.

## Modules

```
src/mtgdata/data.py      download + parquet conversion, card_metadata, DraftTensors / load_tensors
src/mtgdata/games.py     game and replay data (game_path, replay_path, cast_matrix), table helpers
                         (flag, aligned) and shrunk GP / GIH win rates (game_win_rates)
src/mtgdata/cardinfo.py  Scryfall oracle facts per card and deck features built from them
src/mtgdata/encode.py    sparse card bags, passed-pack tokens, others-taken, card_features, draft_context
scripts/download.py      download, convert and sanity-check every set (or pass set codes)
papers/                  literature notes (LITERATURE.md); the PDFs are gitignored
```

```
python scripts/download.py            # all sets below
python scripts/download.py MSH FDN    # just these
```

## Data inventory (downloaded 2026-09-28)

| pack size | picks/draft | sets | wheel-eligible picks |
|---|---|---|---|
| 15 | 45 | LCI | 0–6 |
| 14 | 42 | OTJ, DSK, FDN, DFT, TDM, FIN, SOS, MSH | 0–5 |
| 14 | 41: **P1P1 missing from every draft** | TLA, TMT | 0–5 (skip P1P1) |
| 13 | 39 | MKM, BLB, EOE, ECL | 0–4 |

Premier draft totals: about 1.9M drafts and 76M pick rows, plus about 250k Traditional drafts.
Sampled wheel-subset checks: about 1M pack pairs, 3 violations, all in TradDraft.
Streaming conversion with Int8 columns keeps peak memory around 2.4 GB for each file.


The draft-feature pipelines assume 14-card packs with all 42 picks present, so they only work as-is on the "14, complete" sets.
