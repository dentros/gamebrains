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

## The training transient, which this module used to ignore

**A significance test on the whole match is worthless, and we measured how worthless.** Every
learning agent in a roster typically shares an exploration schedule, so every agent's action
distribution drifts in the same direction over the same rounds whether or not any of them is
reacting to the others. A permutation surrogate cannot null that out, because it destroys the very
temporal structure the shared trend lives in. The result is directed influence detected between
agents that never met.

Measured on this platform, on a null control of agents drawn from *separate* matches, so that every
"significant" pair is a false positive by construction: **35 of 40 pairs flagged, 87.5%, against a
nominal 5%.** Cutting the first half of the series drops it to 47.5%, which is the second half of
the finding: excluding the transient is necessary and not sufficient.

## What replaced the fix, once the fix was tested against ground truth

The first repair here was to derive a burn-in, cut the transient, and refuse to publish unless the
remaining window passed ADF and KPSS. The authors' companion validation study then measured that
procedure over 100 seeds in two games, and it does not survive its own evidence:

- excluding the transient reaches 3.0% in a social dilemma but **11.8% in a coordination game**,
  which is this platform's own congestion family, so exclusion is not sufficient where it matters
  most to us
- it discards roughly **two thirds of every run** to get there
- its apparent margin below nominal in the social dilemma is the estimator's floor rather than
  anything the procedure contributes

What does work is to leave the data alone and change the null instead. Permuting the source
**within blocks of training time** keeps the shared drift inside the null, destroys only the
source-target correspondence, and reaches **5.25% and 5.50%** against a nominal 5% in both games.
It is also the most sensitive remedy tested, detecting 89.0% of injected links in the coordination
game where conditioning on training time detects 61.0% of the same ones.

So this module now does four things, and the first two are the reverse of what it used to do:

1. **Does not cut the data first.** `suggested_burn_in` is still here and still correct, and it is
   off by default.
2. **Permutes the source within blocks** (`transfer_entropy_pairwise(surrogate="blockwise")`),
   which is the default. The unrestricted permutation stays available as `"whole"` so the two can
   be compared on one series.
3. **Gates on the construction's own precondition**, that each marginal is near-constant inside a
   block (`blockwise_precondition`). The companion study measured this failing: driving the
   marginal on a path that moves appreciably within a block takes the rate from 6.50% to 31.75%.
4. **Reports stationarity as a diagnostic, never as a gate.** It says how much drift the surrogate
   is absorbing. A null result under a failed diagnostic is inconclusive, not evidence of
   independence.

The refusal that remains is narrower and better aimed: what is withheld is the single number that
travels into ledgers, figures and papers, and only when the method's stated precondition does not
hold. The per-pair detail is always returned.
"""

from __future__ import annotations

import warnings
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


def predictive_information_per_agent(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """One-step predictive information PI = I(action_{t-1}; action_t) for every agent: how much
    of an agent's own next action is already determined by its own previous one. The "bridge"
    measure of the MI/TE/PI family (see module docstring): a fixed strategy like AllC scores 0
    (its action carries no information because it never varies), an alternator scores ~1 bit, an
    iid random agent ~0. Uses the same lag-1 convention as `transfer_entropy_pairwise` so the two
    are directly comparable (TE measures cross-agent prediction, PI self-prediction, at the same
    timescale)."""
    actions = records["actions"]
    rounds, n_agents = actions.shape
    if rounds < 2:
        return {"bits_by_agent": [0.0] * n_agents, "mean_bits": 0.0, "n_samples": 0,
                "estimator": "discrete plug-in + Miller-Madow correction"}
    per_agent = [_mi_bits(actions[:-1, i], actions[1:, i]) for i in range(n_agents)]
    return {
        "bits_by_agent": per_agent,
        "mean_bits": float(np.mean(per_agent)) if per_agent else 0.0,
        "n_samples": int(rounds - 1),
        "estimator": "discrete plug-in + Miller-Madow correction",
    }


# --- the guardrail -------------------------------------------------------------------------------

#: Significance level for the stationarity diagnostics, and for calling a transfer-entropy pair
#: significant. Kept as one constant because a reader comparing the two should see they agree.
ALPHA = 0.05


def suggested_burn_in(roster: list[Any] | None, rounds: int) -> tuple[int, str]:
    """How many opening rounds to discard, solved from the roster's own exploration schedules.

    An epsilon-greedy schedule states exactly when exploration stops changing: multiplicative decay
    from `epsilon_start` reaches `epsilon_min` after ``log(min/start) / log(decay)`` rounds. Every
    scheduled agent must have got there, so the burn-in is the maximum over the roster.

    This is derived rather than configured on purpose. A constant is wrong for every roster but the
    one it was tuned on, and it is silently wrong, since nothing in the output says which regime the
    numbers came from. Returns the round count and a sentence explaining it, because a discarded
    window that cannot explain itself is indistinguishable from a bug.
    """
    if not roster:
        return 0, ("no roster supplied, so no exploration schedule could be read and no burn-in "
                   "was applied; the estimate covers the whole match including any training "
                   "transient")

    longest = 0
    who = ""
    for agent in roster:
        start = getattr(agent, "epsilon_start", None)
        floor = getattr(agent, "epsilon_min", None)
        decay = getattr(agent, "epsilon_decay", None)
        if start is None or floor is None or decay is None:
            continue
        if not (0.0 < decay < 1.0) or start <= floor or floor <= 0.0:
            continue                      # constant or degenerate schedule: nothing to wait for
        needed = int(np.ceil(np.log(floor / start) / np.log(decay)))
        if needed > longest:
            longest, who = needed, getattr(agent, "name", "?")

    if longest == 0:
        return 0, ("no agent in the roster has a decaying exploration schedule, so there is no "
                   "training transient of this kind to exclude")
    if longest >= rounds:
        return longest, (f"the slowest schedule ({who}) needs {longest:,} rounds to reach its "
                         f"exploration floor and the match is only {rounds:,}, so no part of this "
                         f"run is post-transient")
    return longest, (f"first {longest:,} of {rounds:,} rounds discarded, the point at which the "
                     f"slowest exploration schedule in the roster ({who}) reaches its floor")


def stationarity_report(series: np.ndarray) -> dict[str, Any]:
    """ADF and KPSS on one series, reported together because they test opposite nulls.

    ADF's null is that a unit root is present, so a small p favours stationarity. KPSS's null is
    that the series is stationary, so a small p favours non-stationarity. Running both and
    requiring agreement is stricter than either alone, and the disagreement cases are informative
    rather than embarrassing.

    Note what a pass does and does not mean. Failing to reject a null is not evidence for it, so
    "stationary (both agree)" means the window survived two tests designed to catch the failure
    that matters here, not that stationarity has been established.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        return {"available": False, "verdict": "not tested",
                "note": "statsmodels is not installed, so stationarity could not be checked"}

    x = np.asarray(series, dtype=float)
    if x.size < 20:
        return {"available": True, "verdict": "too short",
                "note": f"{x.size} samples is too few to test"}
    if np.all(x == x[0]):
        return {"available": True, "verdict": "constant",
                "note": "the series never changes, so the question does not arise"}

    with warnings.catch_warnings():
        # KPSS warns whenever its p-value falls outside the published lookup table. That is a
        # statement about table resolution, not about this series, and it fires constantly.
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(x, autolag="AIC")[1])
        kpss_p = float(kpss(x, regression="c", nlags="auto")[1])

    adf_says_stationary = adf_p < ALPHA        # rejects the unit root
    kpss_says_stationary = kpss_p >= ALPHA     # fails to reject stationarity

    if adf_says_stationary and kpss_says_stationary:
        verdict = "stationary"
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive"

    return {"available": True, "verdict": verdict, "adf_p": adf_p, "kpss_p": kpss_p,
            "alpha": ALPHA, "n": int(x.size),
            "note": f"ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}"}


def window_is_trustworthy(records: dict[str, np.ndarray]) -> dict[str, Any]:
    """How much drift is in this window, judged on its worst agent series.

    **Read as a diagnostic.** `compute_all` does not gate on this, because gating on it refuses
    runs the blockwise surrogate handles correctly. What it tells you is how hard the surrogate is
    working, and the companion study found the reading useful precisely because the one game where
    transient exclusion is not enough is also the game with the fewest stationary series.

    One non-stationary series is enough to spoil the pairs it appears in, so the window is judged
    on its worst member rather than on an average.
    """
    actions = records["actions"]
    per_agent = [stationarity_report(actions[:, i]) for i in range(actions.shape[1])]
    verdicts = [r["verdict"] for r in per_agent]

    if "non-stationary" in verdicts:
        ok, why = False, (f"{verdicts.count('non-stationary')} of {len(verdicts)} agent action "
                          f"series are non-stationary")
    elif "inconclusive" in verdicts:
        ok, why = False, (f"{verdicts.count('inconclusive')} of {len(verdicts)} agent action "
                          f"series gave disagreeing stationarity tests")
    elif not per_agent or not per_agent[0].get("available", False):
        ok, why = False, per_agent[0]["note"] if per_agent else "no series to test"
    else:
        ok, why = True, f"all {len(verdicts)} agent action series passed ADF and KPSS"

    return {"ok": ok, "reason": why, "per_agent": per_agent}


#: How the source is permuted to build the null. `"blockwise"` keeps each observation inside its
#: own block of training time; `"whole"` is the unrestricted permutation this module used to use
#: and is kept so the two can be compared on the same series.
SURROGATE_CONSTRUCTIONS = ("blockwise", "whole")

#: Blocks the training-time axis is cut into. The companion study swept 16 to 256 and every
#: setting landed in the nominal region, so this is not a tuned number.
N_BLOCKS = 32

#: The blockwise surrogate's one precondition, as a bound on `within_block_drift_z`.
#:
#: **Measured, and it sits between two cases rather than on a principle.**
#: `experiments/run_te_surrogate_null.py` runs the null control at this platform's own defaults
#: and again under a deliberately fast exploration schedule, over 200 independent pairs each:
#:
#:     epsilon_decay 0.9995 (default)   drift z 3.7 max   blockwise false positives  4.50%
#:     epsilon_decay 0.95   (fast)      drift z 5.2 max   blockwise false positives 10.50%
#:
#: So the statistic separates the configuration the construction handles from the one it does
#: not, and the bound is placed between them. Two calibration points is what that buys and no
#: more: it is a measured separation of two schedules, not a general threshold, and a roster
#: whose drift lands near it deserves the per-pair detail rather than trust in this constant.
#:
#: The first attempt at this bound was a raw marginal shift against a made-up 0.10 cutoff. The
#: null control refuted it on the first run: the raw shift reads 0.40 on series the surrogate
#: handles at the nominal rate, because at a 47-round block that difference is mostly sampling
#: noise. A threshold nobody measured would have refused every default run on this platform.
MAX_WITHIN_BLOCK_DRIFT_Z = 4.5


def _blocks(n: int, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """Contiguous equal-width blocks of training time, the discretisation the surrogate needs."""
    return np.minimum((np.arange(n) * n_blocks) // n, n_blocks - 1).astype(np.int64)


def _block_segments(n: int, n_blocks: int = N_BLOCKS) -> list[np.ndarray]:
    """One index array per block, built once and reused across every pair and every surrogate."""
    b = _blocks(n, n_blocks)
    order = np.argsort(b, kind="stable")
    bounds = np.searchsorted(b[order], np.arange(n_blocks + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]


def within_block_marginal_shift(series: np.ndarray, n_blocks: int = N_BLOCKS) -> float:
    """The largest raw move of a series' mean between the two halves of a single block.

    Named for exactly what it computes, and **kept only for reporting** because on its own it does
    not answer the question it looks like it answers. At this platform's default block length a
    half-block holds about twenty binary samples, so the difference of two such means is dominated
    by sampling noise: the null control measures 0.40 here on series the blockwise surrogate
    handles at 4.5%, nominal. Use `within_block_drift_z` to decide anything.
    """
    b = _blocks(len(series), n_blocks)
    worst = 0.0
    for k in range(n_blocks):
        seg = series[b == k]
        if len(seg) < 4:
            continue
        half = len(seg) // 2
        worst = max(worst, abs(float(seg[:half].mean()) - float(seg[half:].mean())))
    return worst


def within_block_drift_z(series: np.ndarray, n_blocks: int = N_BLOCKS) -> float:
    """The same half-block difference, in units of its own sampling error.

    This is the statistic the precondition is judged on. A raw shift says nothing without knowing
    how large a shift pure noise produces at this block length, and dividing by the binomial
    standard error of the difference supplies exactly that, which also makes the number comparable
    across block lengths and run lengths rather than specific to one configuration.

    Returns the largest absolute z over the blocks. Under no drift this is a maximum of roughly
    `n_blocks` standard normals, so it sits near 2.5 to 3 rather than near 0, and the threshold
    is set from a measured failure case rather than from that expectation.
    """
    b = _blocks(len(series), n_blocks)
    worst = 0.0
    for k in range(n_blocks):
        seg = np.asarray(series[b == k], dtype=float)
        if len(seg) < 8:
            continue
        half = len(seg) // 2
        first, second = seg[:half], seg[half:]
        pooled = float(seg.mean())
        if pooled <= 0.0 or pooled >= 1.0:
            continue                      # a block with no variation carries no drift to measure
        se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / len(first) + 1.0 / len(second)))
        if se <= 0.0:
            continue
        worst = max(worst, abs(float(first.mean()) - float(second.mean())) / float(se))
    return worst


def blockwise_precondition(records: dict[str, np.ndarray],
                           n_blocks: int = N_BLOCKS) -> dict[str, Any]:
    """Is every agent's marginal near-constant inside a block?

    This is the gate. Stationarity is not, any more: the companion study found that treating it
    as one discards runs the construction handles perfectly well, and recommends reading it as a
    diagnostic instead. What does gate is the surrogate's own documented failure mode, because a
    result produced outside a method's stated precondition is not a weak result, it is an
    unsupported one.
    """
    actions = records["actions"]
    rounds, n_agents = actions.shape
    per_block = rounds / n_blocks if n_blocks else 0.0
    zs = [within_block_drift_z(actions[:, i], n_blocks) for i in range(n_agents)]
    shifts = [within_block_marginal_shift(actions[:, i], n_blocks) for i in range(n_agents)]
    worst = max(zs) if zs else 0.0

    if per_block < 8:
        ok, why = False, (f"{rounds:,} rounds over {n_blocks} blocks leaves {per_block:.1f} "
                          f"rounds per block, too few to tell drift from sampling noise inside one")
    elif worst > MAX_WITHIN_BLOCK_DRIFT_Z:
        ok, why = False, (f"an agent's action marginal moves {worst:.1f} standard errors inside a "
                          f"single block, over the {MAX_WITHIN_BLOCK_DRIFT_Z} this construction "
                          f"assumes, so the surrogate is holding constant something that is still "
                          f"changing")
    else:
        ok, why = True, (f"the largest within-block drift of any agent's marginal is {worst:.1f} "
                         f"standard errors, inside the {MAX_WITHIN_BLOCK_DRIFT_Z} this "
                         f"construction assumes")

    return {"ok": ok, "reason": why, "worst_drift_z": worst, "per_agent_drift_z": zs,
            "worst_raw_shift": max(shifts) if shifts else 0.0,
            "n_blocks": n_blocks, "rounds_per_block": per_block}


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
    burn_in: int = 0, surrogate: str = "blockwise", n_blocks: int = N_BLOCKS,
) -> dict[str, Any]:
    """T(agent_i -> agent_j) at lag 1 for every ordered pair, with a surrogate significance test.

    `n_surrogates` permutations of the source's action history are substituted for the real one,
    destroying its correspondence with the target while preserving both agents' marginal action
    statistics, and `p_value` is the fraction of surrogates whose TE meets or exceeds the observed
    value. Raw TE is reported alongside that test and never in place of it: with a finite sample,
    plug-in TE is never exactly zero even between agents with no causal link, so an unvalidated
    point value reads as detected influence when it is sampling noise or a shared confound.

    **How the source is permuted is the whole question**, and the default changed once we had it
    measured. Every learning agent in a roster typically shares an exploration schedule, so every
    action distribution drifts together whether or not anyone is reacting to anyone. An
    unrestricted permutation destroys that drift along with everything else, so the null it builds
    is one in which no drift exists, the observed statistic sits far above it, and agents that
    never met are flagged as influencing one another.

    `"blockwise"`, the default, permutes the source **within** contiguous blocks of training time.
    The drift survives into the null, the source-target correspondence does not, and the series,
    the statistic and the estimand are all left alone. The companion study measured both
    constructions against ground truth over 100 seeds in two games: unrestricted permutation gives
    false-positive rates of 100.00% and 99.95%, blockwise gives 5.25% and 5.50% against a nominal
    5%, and blockwise is also the most sensitive of the remedies tested, detecting 89.0% of
    injected links in the coordination game where conditioning on training time detects 61.0% of
    the same ones. `"whole"` is that unrestricted permutation, kept because comparing the two on
    one series is the only way to see what the construction is doing.

    `burn_in` trims the opening rounds and now defaults to off. Excluding the transient is not
    wrong, but it discards roughly two thirds of a run and it does not reach the size of the test
    in the coordination game, so it is no longer how this module handles the problem. Pass it if
    you are trimming deliberately or comparing against the older behaviour.

    `p_value < ALPHA` flags a pair as significant, uncorrected for the multiple comparisons across
    a roster's pairs. That is a documented simplification rather than an oversight: a small roster
    keeps the pair count low, and a Bonferroni or FDR correction is future work.
    """
    if surrogate not in SURROGATE_CONSTRUCTIONS:
        raise ValueError(f"surrogate must be one of {SURROGATE_CONSTRUCTIONS}, got {surrogate!r}")

    actions = records["actions"]
    if burn_in:
        actions = actions[burn_in:]
    rounds, n_agents = actions.shape
    rng = np.random.default_rng(seed)
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}

    if rounds < 2:
        return {
            "by_pair": {}, "mean_bits": 0.0, "mean_significant_bits": 0.0,
            "n_significant_pairs": 0, "n_pairs": 0,
            "estimator": "discrete plug-in + Miller-Madow correction", "n_surrogates": n_surrogates,
            "surrogate": surrogate,
        }

    # Blocks are a property of the series length alone, so they are built once rather than per
    # pair and per surrogate.
    segments = _block_segments(rounds - 1, n_blocks) if surrogate == "blockwise" else []

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            x_prev = actions[:-1, i]
            y_prev = actions[:-1, j]
            y_t = actions[1:, j]
            observed = _te_bits(x_prev, y_prev, y_t)
            surrogates = np.empty(n_surrogates)
            if surrogate == "blockwise":
                x_s = np.array(x_prev, dtype=np.int64)
                for s in range(n_surrogates):
                    for seg in segments:
                        if len(seg) > 1:
                            x_s[seg] = x_prev[rng.permutation(seg)]
                    surrogates[s] = _te_bits(x_s, y_prev, y_t)
            else:
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
        "surrogate": surrogate,
        "n_blocks": n_blocks if surrogate == "blockwise" else None,
        "burn_in": int(burn_in),
        "rounds_analysed": int(rounds),
    }


def compute_all(records: dict[str, np.ndarray], seed: int = 0, n_surrogates: int = 200,
                roster: list[Any] | None = None, surrogate: str = "blockwise",
                n_blocks: int = N_BLOCKS) -> dict[str, Any]:
    """Entry point mirroring metrics/social.py's `compute_all`.

    The headline scalars are what `_METRIC_META` displays and what a ledger record stores;
    `*_detail` carries the full per-agent and per-pair breakdown for the results page. The
    transfer-entropy headline averages only over pairs that passed the surrogate test, since
    averaging in non-significant pairs would let sampling noise set the number.

    **What gates the headline changed, and the change is the companion study's main finding.**
    This function used to discard the training transient and then refuse to publish unless the
    remaining window passed two stationarity tests. Measured against ground truth, that ordering
    is the wrong one. Exclusion throws away roughly two thirds of a run and still does not reach
    the size of the test in a coordination game, and using stationarity as a gate refuses runs the
    blockwise surrogate handles correctly. So:

      - the source is permuted within blocks of training time, which leaves the series, the
        statistic and the estimand alone and reaches the nominal rate in both games tested
      - the gate is the surrogate's **own** precondition, that each marginal is near-constant
        inside a block, because a result produced outside a method's stated precondition is
        unsupported rather than merely weak
      - stationarity is computed and reported as a **diagnostic**, never as a gate. It says how
        much drift the construction is being asked to absorb
      - burn-in exclusion is available and off by default

    When the precondition fails, `transfer_entropy_bits` is ``None`` and
    `transfer_entropy_bits_status` carries the reason, because a number that is absent without
    explanation is indistinguishable from a defect to whoever reads the record later.

    One reading rule the companion study is explicit about and this docstring repeats because it
    governs how the output is used: **a null result under a failed stationarity diagnostic is
    inconclusive, not evidence of independence.**
    """
    mi = mutual_information_per_agent(records)
    pi = predictive_information_per_agent(records)

    rounds = int(records["actions"].shape[0])
    te = transfer_entropy_pairwise(records, seed=seed, n_surrogates=n_surrogates,
                                   surrogate=surrogate, n_blocks=n_blocks)

    precondition = blockwise_precondition(records, n_blocks) if surrogate == "blockwise" else {
        "ok": False,
        "reason": ("the unrestricted permutation destroys the shared exploration drift along with "
                   "everything else, which is measured at 100.00% and 99.95% false positives, so "
                   "no headline is published from it"),
    }

    # Reported, never gating. It is the answer to "how hard is the surrogate working here?"
    stationarity = window_is_trustworthy(records)
    stationarity["role"] = ("diagnostic, not a gate: a null result under a failed stationarity "
                            "test is inconclusive rather than evidence of independence")
    te["precondition"] = precondition
    te["stationarity"] = stationarity
    te["burn_in_available"] = suggested_burn_in(roster, rounds)[1]

    if not precondition["ok"]:
        headline, status = None, precondition["reason"]
    else:
        headline = te["mean_significant_bits"]
        status = f"{precondition['reason']}; stationarity diagnostic: {stationarity['reason']}"

    return {
        "mutual_information_bits": mi["mean_bits"],
        "mutual_information_detail": mi,
        "transfer_entropy_bits": headline,
        "transfer_entropy_bits_status": status,
        "transfer_entropy_detail": te,
        "predictive_information_bits": pi["mean_bits"],
        "predictive_information_detail": pi,
    }
