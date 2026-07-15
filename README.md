# GameBrains

A platform for **heterogeneous cognitive agents** ("brains") playing **repeated, n-player**
game-theoretic scenarios — with a live console, transparent inspectable internal states, and
social / game-theoretic / information-theoretic metrics. The game engine never knows *how* an
agent decides or learns, so classic strategies, reinforcement learning, active inference, and
evolvable animats can share the same match while each exposes its internal state through one
common interface.

See `CLAUDE.md` for the full internal design brief (architecture rationale, ALT/MBoE reuse map,
per-phase build notes).

## What's here

- **Game:** N-player Public Goods Game (the standard generalization of the Prisoner's Dilemma).
- **Agents** (`agents/`), all sharing one `Agent` contract (`act`/`update`/`inspect`/`render_brain`):
  tabular Q-learning, a Deep Q-Network (PyTorch), a Free-Energy-Principle / active-inference agent
  (Theory-of-Mind-relevant belief tracking), an evolvable Markov-brain animat (genetic algorithm +
  Φ/causal-autonomy via a maintained PyPhi compatibility fork, `vendor/pyphi/`), and classic fixed
  strategies (AllC/AllD/Random/Majority-TFT).
- **Metrics** (`metrics/`): social (cooperation rate, efficiency, payoff Gini, action entropy),
  Nash equilibrium detection (`pygambit`), and Φ/causal autonomy (`vendor/pyphi/`).
- **Interoperability** (`interop/`): thin PettingZoo/Gymnasium adapters — an external RL framework
  (RLlib, Stable-Baselines3, MLPro) can train/evaluate against GameBrains's own cognitive agents
  without either side knowing the other's implementation.
- **Repository-lite** (`repository/`): a local, signed (Ed25519), append-only, content-addressed
  ledger with automatic `extends`/`replicates` lineage detection, plus a Smart Filter (exact +
  k-NN-similarity lookup) — a "blockchain-lite" implementation of the decentralized-repository
  design in `docs/repository-schema.md`. Deliberately local-only for now: no P2P network, no
  Matrix/IPFS layer yet (see that doc's own scope note).
- **Web demo UI** (`webui/`, Flask): a local, form-driven interface to run a match with any roster
  mix and see the live results — brain-creature panels, a cooperation-rate chart, the full
  narrated log, the repository/Smart-Filter record, Nash equilibrium, and Φ/autonomy — plus a
  separate tab for the Markov-brain genetic algorithm with a fair "bake-off" comparison mode
  against pretrained-then-frozen RL agents. A first, functional step toward the platform's
  intended gamified "houses" interface, not that interface itself.

## Run

From the parent directory of `gamebrains/`:

```bash
pip install -r gamebrains/requirements.txt

# CLI: a mixed-roster match, live in the terminal
python -m gamebrains.experiments.run_pgg
python -m gamebrains.experiments.run_pgg --all qlearning --agents 5 --rounds 3000   # free-riding collapse
python -m gamebrains.experiments.run_evolution                                      # evolve Markov-brains

# Web UI: pick every parameter through a browser
python -m gamebrains.webui.app   # -> http://127.0.0.1:5000/
```

Event-logs are written to `gamebrains/results/*.jsonl` (JSON-Lines: a `meta` header, then `round` /
`brain_snapshot` / `generation` events) — the same stream every consumer (terminal console, web UI,
future analyses) reads.

## Tests

```bash
python -m gamebrains.tests.test_repository
python -m gamebrains.tests.test_record
python -m gamebrains.tests.test_smart_filter
python -m gamebrains.tests.test_equilibrium
python -m gamebrains.tests.test_markov_brain
python -m gamebrains.tests.test_evolution
python -m gamebrains.tests.test_pyphi_fork
python -m gamebrains.tests.test_pettingzoo_adapter
python -m gamebrains.tests.test_gym_adapter
```

## License

GPL-3.0 (see `LICENSE`). Two independent reasons: (1) `vendor/pyphi/` is a patched fork of
PyPhi (Mayner et al. 2018), itself GPL-3.0-licensed, so a copyleft license for the repository as a
whole is the correct choice, not just a preference; (2) the authors independently wanted a
"strict but open" license for the rest of the codebase.

Code under `agents/dqn.py`, `agents/qlearning.py` (encoding logic), and `games/` originates from
the authors' own prior, sole-authored ALT/MBoE project (`github.com/dentros/Alternation`) and is
100% author-owned — free to relicense here regardless of that project's own license.

## How to Cite

This is active research software; please cite the most specific publication available at the time:

1. **Preprint** (arXiv) — not yet posted; check back or open an issue.
2. **Conference paper** — *GameBrains: An Open Platform for Comparing Cognitive Agent
   Architectures in Repeated Games* (tool demonstration, CoopIS 2026) — citation details to be
   added once accepted/published.
3. **Journal paper** (systems/software venue, in submission) — citation details to be added once
   accepted/published.

Until all three exist, citing the GitHub repository itself (with commit hash) is acceptable.
