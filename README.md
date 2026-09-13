# GameBrains

A platform for **heterogeneous cognitive agents** ("brains") playing **repeated, n-player**
game-theoretic scenarios, with a live console, transparent inspectable internal states, and
social, game-theoretic and information-theoretic metrics. The game engine never knows *how* an
agent decides or learns, so classic strategies, reinforcement learning, active inference, and
evolvable animats can share the same match while each exposes its internal state through one
common interface.

Matches run either **across different architectures**, which is what the platform is for, or
**across identical ones**, when a single architecture is being studied on its own.

See `CLAUDE.md` for the full internal design brief (architecture rationale, reuse map from the
author's prior alternation-metrics project, per-phase build notes).

## Author

**Nikolaos Al. Papadopoulos**
PhD candidate, Department of Applied Informatics, University of Macedonia, Thessaloniki, Greece

- ORCID: [0000-0003-1842-8227](https://orcid.org/0000-0003-1842-8227)
- Google Scholar: [XkunCRsAAAAJ](https://scholar.google.com/citations?user=XkunCRsAAAAJ)
- GitHub: [@dentros](https://github.com/dentros)
- Email: nikolaos.papadopoulos@uom.edu.gr

Sole author and copyright holder of all code in this repository. See **Authorship** below for how
that relates to co-authorship on the accompanying papers.

## What's here

- **Games** (`games/`): the n-player **Public Goods Game**, the standard generalization of the
  Prisoner's Dilemma, and a parametrised **congestion family** whose headline member is the
  **Honey-Jar Game (HJG)**, an n-player anti-coordination game from the author's prior published
  work. The family also ships three further published variants and a one-shot market-entry game.
- **Agents** (`agents/`), all sharing one `Agent` contract (`act`/`update`/`inspect`/`render_brain`):
  tabular Q-learning, a Deep Q-Network (PyTorch), a Free-Energy-Principle / active-inference agent
  (Theory-of-Mind-relevant belief tracking), an evolvable Markov-brain animat (genetic algorithm
  plus Φ and causal autonomy via a maintained PyPhi compatibility fork, `vendor/pyphi/`), and
  classic fixed strategies (AllC/AllD/Random/Majority-TFT).
- **Metrics** (`metrics/`): social (cooperation rate, efficiency, payoff Gini, action entropy),
  Nash equilibrium detection (`pygambit`), Φ and causal autonomy (`vendor/pyphi/`),
  information-theoretic measures (mutual information, transfer entropy with a blockwise surrogate
  test, predictive information), graph-theoretic measures over the co-cooperation and influence
  networks, and the **ALT and RP temporal-fairness families**, reimplemented from the author's own
  published work and validated against its worked examples.
- **Interoperability** (`interop/`): thin PettingZoo and Gymnasium adapters, plus the shims three
  downstream ecosystems turned out to need. An external RL framework (RLlib, Stable-Baselines3,
  MLPro) can train or evaluate against GameBrains's own cognitive agents without either side
  knowing the other's implementation. All five routes are tested rather than asserted.
- **Repository-lite** (`repository/`): a local, signed (Ed25519), append-only, content-addressed
  ledger with automatic `extends`/`replicates` lineage detection, plus a Smart Filter (exact and
  k-NN-similarity lookup). A "blockchain-lite" implementation of the decentralized-repository
  design in `docs/repository-schema.md`. Deliberately local-only for now: no P2P network, no
  Matrix or IPFS layer yet, and see `repository/ledger.py` for exactly what the signatures and the
  hash chain do and do not establish.
- **Web demo UI** (`webui/`, Flask): a local, form-driven interface to run a match with any roster
  mix and see the results. Brain-creature panels, a cooperation-rate chart, the full narrated log,
  the repository and Smart-Filter record, Nash equilibrium, Φ and autonomy, plus separate tabs for
  the genetic algorithm, cross-run analytics and a parameter-space map. A first, functional step
  toward the platform's intended gamified interface, not that interface itself.

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

Event-logs are written to `gamebrains/results/*.jsonl` (JSON-Lines: a `meta` header, then `round`,
`episode`, `brain_snapshot` and `generation` events). Every consumer reads the same stream, the
terminal console, the web UI and the post-hoc metrics alike.

## Tests

```bash
python -m gamebrains.tests.run_all              # all 23 modules
python -m gamebrains.tests.run_all --skip-slow  # omits the two slow ones
```

Individual modules run the same way, for example `python -m gamebrains.tests.test_repository`.
The interoperability tests skip cleanly when their optional library is absent.

## Documentation

- `docs/adding-a-game.md`, the authoring contract for contributing a game
- `docs/adding-an-agent.md`, the same for an agent
- `docs/repository-schema.md`, the record format and the three identity tiers
- `docs/authoring-trial.md`, the protocol for an external authoring study

## Authorship

All code in this repository was conceived, designed, and implemented by **Nikolaos Al.
Papadopoulos**, who is its sole author and copyright holder. Co-authors on the accompanying papers
(see "How to Cite" below) contributed supervision and manuscript review. They did not author code.

## License

Copyright (C) 2026 Nikolaos Al. Papadopoulos. GPL-3.0 (see `LICENSE`). Two independent reasons:
(1) `vendor/pyphi/` is a patched fork of PyPhi (Mayner et al. 2018), itself GPL-3.0-licensed, so a
copyleft license for the repository as a whole is the correct choice and not merely a preference;
(2) a "strict but open" license was independently wanted for the rest of the codebase.

Code under `agents/dqn.py`, `agents/qlearning.py` (encoding logic), and `games/` originates from
the author's own prior, sole-authored alternation-metrics project
(`github.com/dentros/Alternation`) and is 100% author-owned, free to relicense here regardless of
that project's own license.

## How to Cite

This is active research software. Please cite the most specific publication available at the time:

1. **Preprint** (arXiv), not yet posted for this platform. Single-authored (N. Al. Papadopoulos
   only). Check back or open an issue.
2. **Conference paper**, *GameBrains: A Platform for Cognitive Agent Interaction in Game-Theoretic
   Scenarios*. Citation details to be added once accepted or published. Lists additional co-authors
   for supervision and review, see "Authorship" above.
3. **Journal paper** (software venue, in submission). Citation details to be added once accepted
   or published. Same co-authorship basis as the conference paper.

Until all three exist, citing this repository with a commit hash is acceptable.

The temporal-fairness measures implemented in `metrics/social_alt.py`, and the congestion game
family in `games/congestion.py`, are the author's own prior published work, reimplemented here
rather than devised here. Please cite the originals when those are used:

> N. Al. Papadopoulos, R. Taratori, M. Sanchez-Fibla, and K. E. Psannis, "Rotational Periodicity: a
> scalable metric for turn-taking evaluation in multi-agent systems," in *Proc. 22nd Int. Conf. on
> Modelling Decisions for Artificial Intelligence (MDAI 2025)*, Valencia, Spain, 2025. Springer
> LNCS vol. 15950. [doi:10.1007/978-3-032-03711-4_16](https://doi.org/10.1007/978-3-032-03711-4_16)

> N. Al. Papadopoulos and M. Sanchez-Fibla, "Alternation measures for the evaluation of selfish
> agents' turn-taking," in *Artificial Intelligence Research and Development*, IOS Press, 2021,
> pp. 278-281. [doi:10.3233/FAIA210145](https://doi.org/10.3233/FAIA210145)
