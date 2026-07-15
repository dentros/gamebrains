# GameBrains — Decentralized Repository: record schema & semantics (spec)

Status: **design spec** (not implemented yet). Storage decision: **hybrid** — IPFS (data) + a
signed append-only Merkle ledger (provenance) + Matrix (community). We start with **blockchain-lite
(free, no gas)**; records are chain-agnostic so a Merkle-root can later be anchored to a public L2.

Core principle: **never mutate — add & link.** Every change (extend episodes, add a metric,
replicate with a new seed, publish a meta-analysis) is a *new immutable object* linked into a DAG,
exactly like git. Shared content is stored once (content-addressing) and referenced by many.

---

## 1. What gets stored where

- **IPFS (the data):** the full **experiment package** → addressed by a `content_cid`.
  Because a log file is itself a Merkle-DAG of chunks, a longer run reuses the shorter run's prefix
  chunks and only stores the new tail. We store the **raw event-log**, not just a metrics summary,
  so any future metric is derivable without re-running.
- **Ledger (blockchain-lite):** one small **record** per experiment (below) — pointers + metadata +
  signature + lineage edges. Tiny, immutable, publicly verifiable.
- **Matrix:** notifications, per-experiment rooms, live event-stream. No storage role.

## 2. Experiment package (goes to IPFS → `content_cid`)

```
package/
  manifest.json        # everything in §3 except content_cid (which is this package's own CID)
  event_log.jsonl      # raw per-round events (meta + round + brain_snapshot) — the source of truth
  metrics/             # computed metrics as versioned files (see §5)
    social.v1.json
```

## 3. Ledger record (the small, immutable entry)

```jsonc
{
  "schema": "gamebrains/experiment@1",
  "config_hash":  "…",        // hash of the NORMALIZED config below → EXACT-match dedup key
  "content_cid":  "…",        // IPFS address of the package (§2)

  "game": {                   // normalized config (the thing we hash)
    "name": "public_goods",
    "n_agents": 5,
    "params": { "mpcr": 0.5, "cost": 1.0 }
  },
  "roster": [                 // ordered; drives agent-mix similarity
    { "kind": "qlearning", "params": { "alpha": 0.1, "gamma": 0.95, "eps_decay": 0.9995 } },
    { "kind": "dqn",       "params": { "hidden": 64, "lr": 0.001 } }
  ],
  "horizon":  { "rounds": 5000 },     // the "time" axis (see extension semantics §4)
  "seeds":    { "master": 0 },        // one master seed → deterministic sub-seeds
  "code_version": "git:abcdef1",      // reproducibility anchor

  "metrics_summary": {                // CACHE only; source of truth = raw log in the package
    "cooperation_rate": 0.223, "efficiency": 0.223, "payoff_gini": 0.346, "action_entropy": 0.549
  },
  "feature_vector": [5, 0.5, 1.0, 5000, 2, 1, 0, 2],  // [n, mpcr, cost, rounds, #ql, #dqn, #fep, #classic] → k-NN similarity

  "contributor": "did:key:…",         // public key
  "signature":   "…",                 // signs (config_hash + content_cid + timestamp)
  "timestamp":   "2026-07-05T…",
  "license":     "TBD (code) / CC-BY (data)",

  "parents":  ["<hash of previous ledger record>"],   // Merkle back-link (immutability)
  "lineage":  { "extends": null, "replicates": null, "derives_from": null }  // see §4
}
```

## 4. Relationship semantics (the `lineage` field)

Three ways one experiment relates to another:

| Relationship | When | Effect |
|---|---|---|
| **extends** | same `config_hash` + same `seeds`, larger `horizon.rounds` | The shorter run is a **prefix/superset** of the longer (deterministic). Keep the longer; mark the shorter `superseded_by`. Smart Filter: "need T rounds" is satisfied by any run with `rounds ≥ T` at the same config+seed. **Precondition:** schedules (e.g. epsilon decay) must be **independent of total horizon** — our engine already decays per-step. |
| **replicates** | same `config_hash`, **different** `master` seed | A new sample. Group replications → report mean ± CI. A "result" for a config = the *ensemble* across seeds, not a single run. |
| **derives_from** | new/extra metric, or a meta-analysis | New versioned object pointing to the same `content_cid` (metric) or to several inputs (meta-analysis). Computed **post-hoc / client-side**, no rerun. |

## 5. Metrics are re-derivable (why we keep raw logs)

- Metrics are pure functions of the raw event-log → **compute new metrics later without re-running**,
  over **any episode window** (prefix, `[a,b]`, sliding windows — supports the ALT-style metrics).
- `metrics_summary` in the record is a convenience cache. Adding a metric = a new
  `metrics/<name>.vN.json` in the package (or a `derives_from` record) — never an edit.

## 6. Smart Filter (two lookups over this schema)

1. **Exact** → hash the normalized config, look up `config_hash`. Hit ⇒ fetch `content_cid`, **skip
   rerun** (energy saved). (Cryptographic hash = exact only.)
2. **Similar** → nearest-neighbor on `feature_vector` + faceted/range filters on `game`/`roster` →
   "existing experiments found: X", ranked by closeness, updated in real time as parameters change.

## 7. Open items

- **License** for code (MIT / Apache-2.0 / GPL-3.0) — pending.
- Public-chain **anchoring** cadence (Merkle-root → L2) — later.
- Canonical **normalization** of config before hashing (field order, float rounding) — define before
  implementing dedup.
- Independent seed streams via `numpy.random.SeedSequence(master).spawn(k)`.
