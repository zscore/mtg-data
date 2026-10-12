"""Draft and card encodings shared by the pick, wheel and deck models: sparse card bags, passed-pack
tokens, what the other seven took, card features and per-draft context."""
from __future__ import annotations

import pickle

import numpy as np

from .data import COLORS, FEATURES, N_PACKS, N_SEATS, RANK_ORDER, RARITIES, DraftTensors

CARD_TYPES = ("Creature", "Artifact", "Enchantment", "Instant", "Sorcery", "Land", "Planeswalker",
              "Battle", "Kindred")
SUPERTYPES = ("Legendary", "Basic", "Snow")


def sparse(counts: np.ndarray, k: int):
    """[..., C] card counts -> ([..., k] distinct card indices padded with C, [..., k] copies).

    Keeps card-index order. Stores a pack in k entries instead of C, ~10x less RAM.
    """
    C = counts.shape[-1]
    flat = counts.reshape(-1, C)
    r, c = np.nonzero(flat)
    j = np.arange(len(r)) - np.searchsorted(r, r)  # rank within row (nonzero is row-major)
    assert j.max(initial=0) < k, f"more than {k} distinct cards in a row"
    idx = np.full((len(flat), k), C, np.int16)
    cnt = np.zeros((len(flat), k), np.int8)
    idx[r, j], cnt[r, j] = c, flat[r, c]
    shp = counts.shape[:-1] + (k,)
    return idx.reshape(shp), cnt.reshape(shp)


def pack_tokens(passed: np.ndarray, back: np.ndarray, C: int):
    """Distinct passed cards per (d,k,p), padded with index C. Returns idx, copies, label, mask."""
    idx, copies = sparse(passed, passed.shape[2] + N_SEATS - 1)  # at most n_picks - 1 distinct
    mask = copies > 0
    b = np.concatenate([back, np.zeros(back.shape[:-1] + (1,), back.dtype)], -1)
    label = np.take_along_axis(b, idx.astype(np.int64), -1) > 0
    return idx.astype(np.int64), copies.astype(np.int64), label, mask


def taken_by_others(passed: np.ndarray, back: np.ndarray):
    """Sparse copies the other seven drafters took from the pack that returns at each pick.

    Empty for picks 0..7, whose pack has not been round the table yet.
    """
    D, _, W, C = passed.shape
    k = W + N_SEATS - 1  # 7 normally; malformed records can have more
    idx, cnt = sparse(passed - back, k)
    out_i = np.full((D, N_PACKS, W + N_SEATS, k), C, np.int16)
    out_c = np.zeros(out_i.shape, np.int8)
    out_i[:, :, N_SEATS:], out_c[:, :, N_SEATS:] = idx, cnt
    return out_i, out_c


def type_features(types: list[str]) -> np.ndarray:
    """[C, F] multi-hot of card types and supertypes over all faces; subtypes are ignored.

    "Legendary Artifact Creature - Thopter // Land" -> Legendary, Artifact, Creature, Land.
    """
    main = [{w.replace("Tribal", "Kindred") for f in t.split(" // ") for w in f.split(" - ")[0].split()}
            for t in types]
    cols = [*CARD_TYPES, *SUPERTYPES]
    return np.array([[c in m for c in cols] for m in main], np.float32).reshape(len(types), len(cols))


def card_features(meta, types: bool = True) -> np.ndarray:
    """[C, F] colours, rarity one-hot, mana value, bonus-sheet flag, and either the card types and
    supertypes or just a creature flag."""
    x = [meta.select(list(COLORS)).to_numpy(),
         np.stack([(meta["rarity"] == r).to_numpy() for r in RARITIES], 1),
         meta["mana_value"].clip(0, 8).to_numpy()[:, None] / 8,
         meta.select("is_bonus").to_numpy()]
    if types and "types" in meta.columns:
        x.append(type_features(meta["types"].to_list()))
    else:
        x.append(meta.select("is_creature").to_numpy())
    return np.concatenate([np.asarray(a, np.float32) for a in x], 1)


def draft_context(t: DraftTensors, rank: bool = True, skill: bool = True,
                  time: bool = True) -> np.ndarray:
    """[D, F] per-draft facts the history tokens all get: the drafter's rank, their 17Lands skill
    buckets, and how far into the format the draft happened.

    Time is continuous (days since the set's first draft) rather than a week id: splits are
    chronological, so test drafts fall in weeks a learned per-week embedding never trained on.
    """
    D, x = len(t.draft_id), []
    if rank:  # bronze..mythic, plus unknown (-1)
        x.append(np.eye(len(RANK_ORDER) + 1, dtype=np.float32)[t.rank.astype(np.int64) + 1])
    if skill:  # buckets are -1 when missing
        wr, ng = t.user_wr, t.user_ngames
        x.append(np.stack([np.where(wr >= 0, wr - 0.54, 0) / 0.09, wr < 0,
                           np.log10(ng.clip(min=1)) / 3, ng < 0], 1).astype(np.float32))
    if time:
        ts = np.array(t.draft_time, dtype="datetime64[s]")
        days = (ts - ts.min()).astype(np.float64) / 86400
        x.append(np.stack([days / 30, np.log1p(days) / np.log1p(30)], 1).astype(np.float32))
    return np.concatenate(x, 1) if x else np.zeros((D, 0), np.float32)


class _AnyClassUnpickler(pickle.Unpickler):
    """Reads a pickle even when it names classes from packages that aren't installed (the wheel feature
    cache stores openlane.features.CardStats next to the splits); those load as plain objects."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {})


def splits_for(t: DraftTensors, set_code: str, event: str = "PremierDraft") -> dict[str, np.ndarray]:
    """Train/val/test draft indices: the split saved with the wheel features (FEATURES/<set>.<event>/meta.pkl)
    when it covers these drafts, else chronological 70/15/15. Every model uses the same split."""
    p = FEATURES / f"{set_code}.{event}" / "meta.pkl"
    if p.exists():
        with open(p, "rb") as f:
            sp = _AnyClassUnpickler(f).load()["splits"]
        if sum(len(v) for v in sp.values()) == len(t.draft_id):
            return sp
    D = len(t.draft_id)
    a, b = int(0.7 * D), int(0.85 * D)
    return {"train": np.arange(a), "val": np.arange(a, b), "test": np.arange(b, D)}


def card_inputs(t: DraftTensors, meta: bool, rich: bool, text: bool = False) -> np.ndarray | None:
    """[C, F] per-card model inputs: card_features (meta), Scryfall facts (rich) and the rules-text
    embedding from picknet's scripts/card_text_embeddings.py (text). None if all are off."""
    parts = []
    if meta:
        parts.append(card_features(t.meta))
    if rich:
        from .cardinfo import card_info, card_matrix
        parts.append(card_matrix(card_info(t.names, t.set_code)))
    if text:
        parts.append(np.load(FEATURES / f"{t.set_code}.PremierDraft" / "text_emb.npy"))
    return np.hstack(parts).astype(np.float32) if parts else None
