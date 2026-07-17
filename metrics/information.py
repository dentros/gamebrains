"""
Mutual information and transfer entropy for the Public Goods Game (CLAUDE.md section 5's
Theory-of-Mind-relevant metric family: MI/TE quantify whether one agent's behavior is
statistically explained by, or predictive of, another signal in the match).

Operates on the same per-round `records` dict produced by engine.runner.run_match and consumed by
metrics/social.py (`actions`: (rounds, n_agents) int array; `cooperators`: (rounds,) int array).
No engine changes were needed: PublicGoodsGame.step() broadcasts the identical `k_prev` observation
to every agent (public information, see games/public_goods.py), so that shared observation series
is reconstructed here directly from `cooperators`, one round shifted.

Both metrics use a discrete plug-in (maximum-likelihood) entropy estimator with a Miller-Madow bias
correction -- appropriate because every symbol involved (observations, actions) is already a small
discrete integer in this game, so no continuous-valued estimator (KSG) or neural training surrogate
(MINE/InfoNCE) is needed. Per the project's own information-theoretic guardrails
(arXiv:2604.23716, cited in CLAUDE.md section 5): this is a calibrated *estimator*, not a training
surrogate, and every reported transfer-entropy value ships with its estimator parameters
(alphabet, sample size) and a surrogate-based significance test -- never a bare point number.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _entropy_bits_mm(symbols: np.ndarray) -> float:
    """Miller-Madow-corrected discrete plug-in entropy, in bits.

    The plug-in estimator H_hat = -sum(p_hat * log2(p_hat)) is negatively biased for finite
    samples; Miller-Madow adds the standard first-order correction (K-1)/(2*N*ln 2), K = number of
    distinct symbols actually observed, N = sample size. `symbols` must be non-negative ints
    (a `np.bincount` requirement) -- every caller below encodes its alphabet that way.
    """
    n = symbols.size
    counts = np.bincount(symbols)
    counts = counts[counts > 0]
    p = counts / n
    h_plugin = float(-np.sum(p * np.log2(p)))
    k = counts.size
    correction = (k - 1) / (2.0 * n * np.log(2))
    return h_plugin + correction


def _reconstruct_observations(cooperators: np.ndarray, n_agents: int) -> np.ndarray:
    """The shared pre-action observation every agent actually saw each round (previous round's
    cooperator count, or the `start` sentinel for round 0) -- recovered purely from the logged
    `cooperators` series, valid because the observation is public/identical for every agent (see
    module docstring), so no engine change is needed to recover it after a match has already run.
    """
    rounds = cooperators.shape[0]
    obs = np.empty(rounds, dtype=np.int64)
    obs[0] = n_agents + 1  # matches PublicGoodsGame.start_state
    if rounds > 1:
        obs[1:] = cooperators[:-1]
    return obs


def _mi_bits(x: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) in bits, via H(X) + H(Y) - H(X,Y), each term Miller-Madow-corrected. `y` must be
    non-negative ints; its observed cardinality is used to build a unique joint symbol for (x, y).
    """
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    y_card = int(y.max()) + 1
    joint = x * y_card + y
    return _entropy_bits_mm(x) + _entropy_bits_mm(y) - _entropy_bits_mm(joint)


def mutual_information_per_agent(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """I(observation; action) for every agent: does the public previous-round signal statistically
    explain this agent's current action? A near-zero value means the agent's action looks
    independent of that signal (e.g. a fixed strategy, or an online learner that hasn't picked up
    on it yet); a value near the action's own entropy means the action is (close to) a deterministic
    function of the observation.
    """
    actions = records["actions"]
    cooperators = records["cooperators"]
    rounds, n_agents = actions.shape
    obs = _reconstruct_observations(cooperators, n_agents)
    per_agent = [_mi_bits(obs, actions[:, i]) for i in range(n_agents)]
    return {
        "bits_by_agent": per_agent,
        "mean_bits": float(np.mean(per_agent)) if per_agent else 0.0,
        "n_samples": int(rounds),
        "estimator": "discrete plug-in + Miller-Madow correction",
    }


def _te_bits(x_prev: np.ndarray, y_prev: np.ndarray, y_t: np.ndarray) -> float:
    """T(X->Y) at lag 1 = I(Y_t ; X_prev | Y_prev), via the standard conditional-MI identity
    T = H(Y_t,Y_prev) + H(X_prev,Y_prev) - H(Y_prev) - H(Y_t,X_prev,Y_prev). All three inputs are
    binary action symbols (0/1) here, so each joint alphabet is fixed and small (<=8 symbols for
    the triple) -- encoded directly rather than through a generic mixed-radix helper.
    """
    x_prev = x_prev.astype(np.int64)
    y_prev = y_prev.astype(np.int64)
    y_t = y_t.astype(np.int64)
    h_y_prev = _entropy_bits_mm(y_prev)
    h_yt_yprev = _entropy_bits_mm(y_t * 2 + y_prev)
    h_xprev_yprev = _entropy_bits_mm(x_prev * 2 + y_prev)
    h_yt_xprev_yprev = _entropy_bits_mm(y_t * 4 + x_prev * 2 + y_prev)
    return h_yt_yprev + h_xprev_yprev - h_y_prev - h_yt_xprev_yprev


def transfer_entropy_pairwise(
    records: dict[str, np.ndarray], seed: int = 0, n_surrogates: int = 200,
) -> dict[str, Any]:
    """T(agent_i -> agent_j) at lag 1 for every ordered pair, each with a time-shift surrogate
    significance test: `n_surrogates` independent random permutations of the source's action
    history are substituted in place of the real one (destroying any genuine temporal link to the
    target while preserving both agents' own marginal action statistics), and `p_value` is the
    fraction of those surrogates whose TE meets or exceeds the real, observed TE. Raw TE is
    reported alongside this test, never in place of it: with a finite sample, plug-in TE is never
    exactly zero even between two agents with no real causal link, so an unvalidated point value
    would silently read as a "detected" influence that is actually sampling noise or a shared
    confound (e.g. both agents reacting to the same public signal).

    `p_value < 0.05` is used below to flag a pair as "significant" -- uncorrected for the multiple
    comparisons across all pairs in a roster, a documented simplification (a small roster keeps
    the pair count low; a full Bonferroni/FDR correction is future work, not hidden here).
    """
    actions = records["actions"]
    rounds, n_agents = actions.shape
    rng = np.random.default_rng(seed)
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}

    if rounds < 2:
        return {
            "by_pair": {}, "mean_bits": 0.0, "mean_significant_bits": 0.0,
            "n_significant_pairs": 0, "n_pairs": 0,
            "estimator": "discrete plug-in + Miller-Madow correction", "n_surrogates": n_surrogates,
        }

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            x_prev = actions[:-1, i]
            y_prev = actions[:-1, j]
            y_t = actions[1:, j]
            observed = _te_bits(x_prev, y_prev, y_t)
            surrogates = np.empty(n_surrogates)
            for s in range(n_surrogates):
                surrogates[s] = _te_bits(rng.permutation(x_prev), y_prev, y_t)
            p_value = float((np.sum(surrogates >= observed) + 1) / (n_surrogates + 1))
            by_pair[(i, j)] = {
                "bits": observed, "p_value": p_value, "n_surrogates": n_surrogates,
                "surrogate_mean_bits": float(surrogates.mean()),
                "surrogate_std_bits": float(surrogates.std()),
            }

    all_bits = [r["bits"] for r in by_pair.values()]
    sig_bits = [r["bits"] for r in by_pair.values() if r["p_value"] < 0.05]
    return {
        "by_pair": by_pair,
        "mean_bits": float(np.mean(all_bits)) if all_bits else 0.0,
        "mean_significant_bits": float(np.mean(sig_bits)) if sig_bits else 0.0,
        "n_significant_pairs": len(sig_bits),
        "n_pairs": len(by_pair),
        "estimator": "discrete plug-in + Miller-Madow correction",
        "n_surrogates": n_surrogates,
    }


def compute_all(records: dict[str, np.ndarray], seed: int = 0, n_surrogates: int = 200) -> dict[str, Any]:
    """Entry point mirroring metrics/social.py's `compute_all`. The two headline scalars
    (`mutual_information_bits`, `transfer_entropy_bits`) are what `_METRIC_META` displays;
    `*_detail` carries the full per-agent / per-pair breakdown (including every TE p-value) for
    the results page's dedicated information-theory panel. `transfer_entropy_bits` deliberately
    averages only over pairs that passed the surrogate significance test (0.0 if none did) --
    averaging in non-significant pairs would let sampling noise dominate the headline number,
    exactly what the surrogate test above exists to catch.
    """
    mi = mutual_information_per_agent(records)
    te = transfer_entropy_pairwise(records, seed=seed, n_surrogates=n_surrogates)
    return {
        "mutual_information_bits": mi["mean_bits"],
        "mutual_information_detail": mi,
        "transfer_entropy_bits": te["mean_significant_bits"],
        "transfer_entropy_detail": te,
    }
