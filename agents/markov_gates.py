"""
Markov brain built from logic gates, as the literature defines one (kind ``"markov_gates"``).

In the work this platform cites (Edlund et al. 2011, Hintze et al. 2017, Albantakis et al. 2014;
the MABE framework), a Markov brain is a network of binary nodes updated by **logic gates**, each
gate a small probabilistic or deterministic lookup table. `agents/markov_brain.py` is something
else: a stochastic recurrent network of sigmoid units, `P(node = 1) = sigmoid(W . state + b)`.
That agent stays exactly as it is, under its own kind, because experiments and records already
name it. This module is the gate-based one, kept alongside rather than in its place.

**Nodes.** The same layout and encoding as `markov_brain.py`, so the two play the same games:
`[sensor_0..sensor_{ns-1}, hidden_0..hidden_{nh-1}, motor_0..motor_{nm-1}]`, sensor `k` is bit `k`
of the observation index, and the action is the motor bits read little-endian. The brain clears
its nodes at the start of every match and whenever it sees the game's start observation, as the
sigmoid brain does. A game that declares no start observation (the congestion family) is given
`NO_START_OBSERVATION`, which no observation equals, so memory runs across the episodes of one
match: carrying who won last time is how a brain could come to take turns. The sigmoid brain's
default, `n_states - 1`, is a real board position in that family, so it is not used there.

**Gates, after MABE.** Each gate has 1 to 4 distinct input node ids (any node) and 1 to 4 distinct
output node ids (hidden or motor nodes only, since sensors are overwritten by the world), and a
table of shape `2^n_in x 2^n_out` whose rows are distributions over output patterns. Both indices
are little-endian over the gate's own ordering: input `k` of the gate contributes bit `k` of the
row index, and bit `k` of the chosen column is written to output `k`. One update is

    1. sensors are clamped from the observation (the state at t);
    2. every gate reads the state at t and draws an output pattern from the row its inputs select;
    3. hidden and motor nodes start from zero at t+1, and each gate's output bits are OR'ed in.

A node that no gate writes is therefore 0 at t+1, and two gates writing one node combine by OR,
which is MABE's rule. Gates read one snapshot of the state, so their order in the list never
changes what they compute.

**Deterministic by default.** Every row is one-hot, as in Albantakis et al.'s animats, so no random
number is drawn during play and `full_tpm()` is the exact transition function, which is what Φ and
causal autonomy want. `deterministic=False` gives probabilistic gates instead, drawing one number
per gate per update from the agent's own seeded generator.

**What the state-by-node TPM cannot hold for probabilistic gates.** `full_tpm()` returns
`P(node_i = 1 | state)` per node, the form PyPhi and `metrics/phi_autonomy.py` consume. That form
treats the nodes as independent given the state. Gates do sample independently of each other, so
each node's marginal is exact, but one probabilistic gate with several outputs draws them jointly,
and the correlation between its own outputs is lost. `tpm_is_exact()` says whether that loss
occurs for this genome: it is True for deterministic brains and for any probabilistic gate whose
rows factorise over their output bits (a single-output gate always does). The causal-autonomy
measure has the same split: its second term is zero for deterministic dynamics, so on a
deterministic gate brain the second term of causal autonomy is zero, and `phi_autonomy` computes
both terms for either brain.

**Genome and the GA.** The gate list is the genome. `spawn`, `crossover`, `clone` and `mutate` have
the signatures `engine.evolution` expects and `markov_brain.py` uses. Mutation does point mutation
of table rows (renormalised), rewiring of one input or output, and gate insertion, deletion and
duplication within `[MIN_GATES, MAX_GATES]`. Crossover cuts each parent's gate list at its own
point and joins the two pieces. Every operator returns a genome that passes the same validation
the constructor applies, so node ids stay in range and every table keeps its shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..engine.agent import Agent
# Both Markov brains read the start observation the same way, so the helper lives with the
# Game contract. Still importable from here, where the tests and the factory first found it.
from ..engine.game import NO_START_OBSERVATION, start_observation  # noqa: F401

#: MABE's bounds on gate arity.
MAX_GATE_INPUTS = 4
MAX_GATE_OUTPUTS = 4
#: Bounds on the genome's length. Insertion and duplication stop at the top, deletion at the bottom,
#: and crossover clips its child into the same range.
MIN_GATES = 1
MAX_GATES = 32
#: Gates in a freshly spawned brain when nothing else is asked for.
DEFAULT_N_GATES = 4

#: What `spawn` reads from `EvolutionConfig.genome`. `n_hidden` comes from the config's own field,
#: as it does for the sigmoid brain.
GENOME_KNOBS = ("n_gates", "deterministic")

_ROW_SUM_TOL = 1e-9


def parse_deterministic(value: Any) -> bool:
    """`deterministic` as a bool, from a bool, 0/1 or the words a form or a sweep may send.

    Anything else is refused. `bool("0")` is True, so a plain cast would quietly turn a request
    for probabilistic gates into deterministic ones, and the run would be recorded as asked for.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("0", "1", "true", "false"):
        return value.strip().lower() in ("1", "true")
    raise ValueError(f"deterministic must be true or false (or 1 or 0), got {value!r}")


def _bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> k) & 1 for k in range(width))


@dataclass
class Gate:
    """One logic gate: which nodes it reads, which it writes, and its table.

    `table[r, c]` is the probability that the gate writes output pattern `c` when its inputs read
    pattern `r`; both are little-endian over this gate's own `inputs` and `outputs` order.
    """

    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    table: np.ndarray

    @property
    def n_in(self) -> int:
        return len(self.inputs)

    @property
    def n_out(self) -> int:
        return len(self.outputs)

    def copy(self) -> "Gate":
        return Gate(tuple(self.inputs), tuple(self.outputs), self.table.copy())

    def row_index(self, state: np.ndarray) -> int:
        return sum(int(state[node]) << k for k, node in enumerate(self.inputs))

    def p_one(self, row: int) -> np.ndarray:
        """P(output k = 1) for each of this gate's outputs, given input pattern `row`."""
        cols = np.arange(self.table.shape[1])
        return np.array([self.table[row, ((cols >> k) & 1) == 1].sum() for k in range(self.n_out)])

    def factorises(self, tol: float = 1e-9) -> bool:
        """Whether every row is the product of its output marginals, so that writing the gate's
        outputs as independent per-node probabilities loses nothing."""
        if self.n_out == 1:
            return True
        cols = np.arange(self.table.shape[1])
        for r in range(self.table.shape[0]):
            p1 = self.p_one(r)
            product = np.ones(len(cols))
            for k in range(self.n_out):
                on = ((cols >> k) & 1) == 1
                product *= np.where(on, p1[k], 1.0 - p1[k])
            if np.max(np.abs(product - self.table[r])) > tol:
                return False
        return True


#: Names for the deterministic gates a reader will recognise, keyed by the output column of each
#: row in row order. Row order is little-endian over the inputs, so for inputs (a, b) the rows are
#: (a,b) = (0,0), (1,0), (0,1), (1,1). Gates outside this set are shown by their table alone.
_ONE_INPUT_NAMES = {(0, 1): "COPY", (1, 0): "NOT", (0, 0): "ZERO", (1, 1): "ONE"}
_TWO_INPUT_NAMES = {(0, 0, 0, 1): "AND", (0, 1, 1, 1): "OR", (0, 1, 1, 0): "XOR",
                    (1, 1, 1, 0): "NAND", (1, 0, 0, 0): "NOR", (1, 0, 0, 1): "XNOR",
                    (0, 0, 0, 0): "ZERO", (1, 1, 1, 1): "ONE"}


def _random_gate(rng: np.random.Generator, n_nodes: int, n_sensor: int,
                 deterministic: bool) -> Gate:
    """A gate with random arity, wiring and table, drawn from `rng` only."""
    n_controlled = n_nodes - n_sensor
    n_in = int(rng.integers(1, min(MAX_GATE_INPUTS, n_nodes) + 1))
    n_out = int(rng.integers(1, min(MAX_GATE_OUTPUTS, n_controlled) + 1))
    inputs = tuple(int(i) for i in rng.choice(n_nodes, size=n_in, replace=False))
    outputs = tuple(n_sensor + int(i) for i in rng.choice(n_controlled, size=n_out, replace=False))
    rows, cols = 2 ** n_in, 2 ** n_out
    if deterministic:
        table = np.zeros((rows, cols))
        table[np.arange(rows), rng.integers(0, cols, size=rows)] = 1.0
    else:
        table = rng.random((rows, cols))
        table /= table.sum(axis=1, keepdims=True)
    return Gate(inputs, outputs, table)


class MarkovGatesAgent(Agent):
    kind = "markov_gates"
    training_mode = "evolutionary"
    # Motor bits map to action indices positionally and selection decides which index pays, so
    # the gates interpret nothing, exactly as for the sigmoid brain.
    semantics = "index-agnostic"
    #: Hidden nodes carry state from one round to the next, bounded by how few there are.
    information = "observation+memory"

    def __init__(
        self,
        name: str,
        n_states: int,
        n_actions: int,
        n_hidden: int = 2,
        n_gates: int = DEFAULT_N_GATES,
        deterministic: bool = True,
        gates: Optional[Sequence[Gate]] = None,
        seed: int = 0,
        start_state: Optional[int] = None,
    ) -> None:
        self.name = name
        self.n_sensor = max(1, (n_states - 1).bit_length())
        self.n_motor = max(1, (n_actions - 1).bit_length())
        if n_actions != 2 ** self.n_motor:
            raise ValueError("MarkovGatesAgent requires n_actions to be a power of 2 "
                             f"(got n_actions={n_actions})")
        if int(n_hidden) < 0:
            raise ValueError(f"n_hidden must be 0 or more, got {n_hidden}")
        self.n_hidden = int(n_hidden)
        self.n_nodes = self.n_sensor + self.n_hidden + self.n_motor
        self.n_controlled = self.n_hidden + self.n_motor
        self.deterministic = parse_deterministic(deterministic)

        self.rng = np.random.default_rng(seed)
        self.start_state = start_state if start_state is not None else n_states - 1

        if gates is None:
            if not MIN_GATES <= int(n_gates) <= MAX_GATES:
                raise ValueError(f"n_gates must be between {MIN_GATES} and {MAX_GATES}, "
                                 f"got {n_gates}")
            gates = [_random_gate(self.rng, self.n_nodes, self.n_sensor, self.deterministic)
                     for _ in range(int(n_gates))]
        self.gates: list[Gate] = [
            Gate(tuple(int(i) for i in g.inputs), tuple(int(o) for o in g.outputs),
                 np.array(g.table, dtype=float))
            for g in gates
        ]
        self.validate()
        self.state = np.zeros(self.n_nodes, dtype=int)

    # --- genome validity ---------------------------------------------------------------------

    @property
    def n_gates(self) -> int:
        """How many gates this genome has now. Recorded, so an evolved brain reports its own."""
        return len(self.gates)

    def validate(self) -> None:
        """Refuse a genome that names a node it does not have or a table of the wrong shape.

        Run by the constructor, so every operator below, which builds its result through it, can
        only return a valid genome or raise.
        """
        if not MIN_GATES <= len(self.gates) <= MAX_GATES:
            raise ValueError(f"a gate brain has between {MIN_GATES} and {MAX_GATES} gates, "
                             f"this one has {len(self.gates)}")
        for g_idx, gate in enumerate(self.gates):
            where = f"gate {g_idx}"
            if not 1 <= gate.n_in <= MAX_GATE_INPUTS:
                raise ValueError(f"{where} has {gate.n_in} inputs, outside 1..{MAX_GATE_INPUTS}")
            if not 1 <= gate.n_out <= MAX_GATE_OUTPUTS:
                raise ValueError(f"{where} has {gate.n_out} outputs, outside 1..{MAX_GATE_OUTPUTS}")
            if len(set(gate.inputs)) != gate.n_in or len(set(gate.outputs)) != gate.n_out:
                raise ValueError(f"{where} lists a node twice: inputs {gate.inputs}, "
                                 f"outputs {gate.outputs}")
            if any(not 0 <= i < self.n_nodes for i in gate.inputs):
                raise ValueError(f"{where} reads node ids {gate.inputs}, but the brain has nodes "
                                 f"0..{self.n_nodes - 1}")
            if any(not self.n_sensor <= o < self.n_nodes for o in gate.outputs):
                raise ValueError(f"{where} writes node ids {gate.outputs}, but only hidden and "
                                 f"motor nodes {self.n_sensor}..{self.n_nodes - 1} can be written")
            shape = (2 ** gate.n_in, 2 ** gate.n_out)
            if gate.table.shape != shape:
                raise ValueError(f"{where} has a table of shape {gate.table.shape}, expected "
                                 f"{shape} for {gate.n_in} inputs and {gate.n_out} outputs")
            if not np.all(np.isfinite(gate.table)) or np.any(gate.table < 0):
                raise ValueError(f"{where} has a negative or non-finite table entry")
            if np.max(np.abs(gate.table.sum(axis=1) - 1.0)) > _ROW_SUM_TOL:
                raise ValueError(f"{where} has a row that does not sum to 1")
            if self.deterministic and not (np.all((gate.table == 0.0) | (gate.table == 1.0))):
                raise ValueError(f"{where} has a row that is not one-hot, in a brain declared "
                                 f"deterministic")

    def tpm_is_exact(self) -> bool:
        """Whether `full_tpm()` describes this brain's dynamics without loss.

        Always True for deterministic gates. For probabilistic ones it is True when every gate's
        rows factorise over their output bits; otherwise the TPM keeps every node's marginal but
        drops the correlation between outputs one gate draws together.
        """
        return self.deterministic or all(g.factorises() for g in self.gates)

    # --- dynamics -----------------------------------------------------------------------------

    def _encode_observation(self, observation: int) -> np.ndarray:
        return np.array([(observation >> k) & 1 for k in range(self.n_sensor)], dtype=int)

    def _decode_action(self) -> int:
        motor_bits = self.state[self.n_sensor + self.n_hidden:]
        return int(sum(int(b) << k for k, b in enumerate(motor_bits)))

    def _draw(self, gate: Gate, row: int) -> int:
        """The output column a gate writes for input pattern `row`."""
        if self.deterministic:
            return int(np.argmax(gate.table[row]))
        cumulative = np.cumsum(gate.table[row])
        column = int(np.searchsorted(cumulative, self.rng.random(), side="right"))
        return min(column, gate.table.shape[1] - 1)

    def _step(self, state: np.ndarray) -> np.ndarray:
        """Hidden and motor nodes at t+1 from the full state at t: zero, then every gate OR'ed in."""
        nxt = np.zeros(self.n_nodes, dtype=int)
        for gate in self.gates:
            column = self._draw(gate, gate.row_index(state))
            for k, node in enumerate(gate.outputs):
                if (column >> k) & 1:
                    nxt[node] = 1
        return nxt[self.n_sensor:]

    def reset_state(self) -> None:
        self.state = np.zeros(self.n_nodes, dtype=int)

    def on_match_start(self, game: Any) -> None:
        # A match starts from zero whatever the previous one left behind. Asks the game nothing
        # about meaning, which is what `index-agnostic` declares.
        self.reset_state()

    def act(self, observation: int) -> int:
        if observation == self.start_state:
            self.reset_state()
        self.state[:self.n_sensor] = self._encode_observation(observation)
        self.state[self.n_sensor:] = self._step(self.state)
        return self._decode_action()

    def update(self, observation: int, action: int, reward: float,
               next_observation: int, done: bool) -> None:
        return None  # evolutionary: the genome does not change within a match

    # --- genome / GA hooks --------------------------------------------------------------------

    def _with_gates(self, gates: Sequence[Gate], name: str, seed: int) -> "MarkovGatesAgent":
        return MarkovGatesAgent(
            name=name, n_states=2 ** self.n_sensor, n_actions=2 ** self.n_motor,
            n_hidden=self.n_hidden, deterministic=self.deterministic, gates=gates, seed=seed,
            start_state=self.start_state,
        )

    def clone(self, name: Optional[str] = None, seed: Optional[int] = None) -> "MarkovGatesAgent":
        """The identical genome with a reset state and (by default) a new generator."""
        new_seed = seed if seed is not None else int(self.rng.integers(1 << 30))
        return self._with_gates([g.copy() for g in self.gates], name or self.name, new_seed)

    def mutate(self, rng: np.random.Generator, rate: float = 0.15, scale: float = 0.4,
               name: Optional[str] = None, seed: int = 0) -> "MarkovGatesAgent":
        """A mutated copy. `rng` drives every choice, so the GA run reproduces; `seed` seeds the
        child's own generator.

        Per gate, each table row mutates with probability `rate`: a deterministic row moves its
        one to a different output pattern, a probabilistic row has one entry moved by a Gaussian
        of width `scale`, clipped at zero and renormalised. Each gate is rewired with probability
        `rate`, one input or one output moved to a node that gate does not already use. Then the
        list itself: one gate deleted with probability `rate`, one inserted at random and one
        duplicated each with probability `rate / 2`, so growth and shrinkage balance in
        expectation away from the bounds. At `rate = 0` the child is the parent's genome.
        """
        gates = [g.copy() for g in self.gates]
        for gate in gates:
            rows, cols = gate.table.shape
            for r in np.flatnonzero(rng.random(rows) < rate):
                if self.deterministic:
                    current = int(np.argmax(gate.table[r]))
                    others = [c for c in range(cols) if c != current]
                    gate.table[r] = 0.0
                    gate.table[r, others[int(rng.integers(len(others)))]] = 1.0
                else:
                    j = int(rng.integers(cols))
                    gate.table[r, j] = max(0.0, gate.table[r, j] + rng.normal(scale=scale))
                    total = gate.table[r].sum()
                    gate.table[r] = gate.table[r] / total if total > 0 else 1.0 / cols
            if rng.random() < rate:
                self._rewire(gate, rng)

        if len(gates) > MIN_GATES and rng.random() < rate:
            del gates[int(rng.integers(len(gates)))]
        if len(gates) < MAX_GATES and rng.random() < rate / 2:
            gates.append(_random_gate(rng, self.n_nodes, self.n_sensor, self.deterministic))
        if len(gates) < MAX_GATES and rng.random() < rate / 2:
            gates.append(gates[int(rng.integers(len(gates)))].copy())
        return self._with_gates(gates, name or self.name, seed)

    def _rewire(self, gate: Gate, rng: np.random.Generator) -> None:
        """Move one input or one output of `gate` to a node it does not already use, if any."""
        rewire_input = rng.random() < 0.5
        if rewire_input:
            current, pool = list(gate.inputs), range(self.n_nodes)
        else:
            current, pool = list(gate.outputs), range(self.n_sensor, self.n_nodes)
        position = int(rng.integers(len(current)))
        free = [n for n in pool if n not in current]
        if not free:
            return
        current[position] = free[int(rng.integers(len(free)))]
        if rewire_input:
            gate.inputs = tuple(current)
        else:
            gate.outputs = tuple(current)

    @classmethod
    def random(cls, name: str, n_states: int, n_actions: int, n_hidden: int = 2,
               n_gates: int = DEFAULT_N_GATES, deterministic: bool = True, seed: int = 0,
               start_state: Optional[int] = None) -> "MarkovGatesAgent":
        return cls(name, n_states, n_actions, n_hidden=n_hidden, n_gates=n_gates,
                   deterministic=deterministic, seed=seed, start_state=start_state)

    # --- introspection ------------------------------------------------------------------------

    def full_tpm(self) -> np.ndarray:
        """State-by-node TPM: `tpm[s, i] = P(node i = 1 at t+1 | joint state s at t)`.

        Row `s` decodes to a state little-endian, `state[k] = (s >> k) & 1`, the convention of
        `pyphi.convert.le_index2state` and of `MarkovBrainAgent.full_tpm`. Sensor nodes copy
        themselves, the standard treatment of clamped inputs. A hidden or motor node is 1 unless
        every gate writing it writes 0, and gates draw independently, so
        `P(node = 1) = 1 - prod_g (1 - P_g(bit = 1))`.

        **Exact for deterministic gates**, where every entry is 0 or 1. For probabilistic gates the
        per-node form loses the correlation between outputs that one gate draws together, so check
        `tpm_is_exact()` before reading a Φ computed from it as exact.
        """
        n = self.n_nodes
        tpm = np.zeros((2 ** n, n))
        for s in range(2 ** n):
            state = np.array(_bits(s, n), dtype=int)
            tpm[s, :self.n_sensor] = state[:self.n_sensor]
            p_zero = np.ones(n)
            for gate in self.gates:
                p_one = gate.p_one(gate.row_index(state))
                for k, node in enumerate(gate.outputs):
                    p_zero[node] *= 1.0 - p_one[k]
            tpm[s, self.n_sensor:] = 1.0 - p_zero[self.n_sensor:]
        return tpm

    def node_labels(self) -> list[str]:
        return ([f"s{k}" for k in range(self.n_sensor)]
                + [f"h{k}" for k in range(self.n_hidden)]
                + [f"m{k}" for k in range(self.n_motor)])

    def _gate_payload(self, gate: Gate, labels: list[str]) -> dict[str, Any]:
        """One gate for a reader: its wiring and its table, as a truth table when deterministic
        and as the likeliest output per row when not."""
        payload: dict[str, Any] = {
            "inputs": list(gate.inputs), "outputs": list(gate.outputs),
            "input_labels": [labels[i] for i in gate.inputs],
            "output_labels": [labels[o] for o in gate.outputs],
        }
        rows = gate.table.shape[0]
        in_bits = ["".join(str(b) for b in _bits(r, gate.n_in)) for r in range(rows)]
        if self.deterministic:
            columns = [int(np.argmax(gate.table[r])) for r in range(rows)]
            payload["truth_table"] = [
                [in_bits[r], "".join(str(b) for b in _bits(columns[r], gate.n_out))]
                for r in range(rows)]
            name = None
            if gate.n_out == 1:
                name = {1: _ONE_INPUT_NAMES, 2: _TWO_INPUT_NAMES}.get(gate.n_in, {}).get(
                    tuple(columns))
            payload["function"] = name
        else:
            summary = []
            for r in range(rows):
                row = gate.table[r]
                best = int(np.argmax(row))
                nz = row[row > 0]
                summary.append({
                    "in": in_bits[r],
                    "likeliest": "".join(str(b) for b in _bits(best, gate.n_out)),
                    "p": float(row[best]),
                    "entropy_bits": float(-(nz * np.log2(nz)).sum()),
                })
            payload["summary"] = summary
        return payload

    def render_brain(self) -> dict:
        labels = self.node_labels()
        return {
            "kind": self.kind,
            "n_sensor": self.n_sensor, "n_hidden": self.n_hidden, "n_motor": self.n_motor,
            "state": self.state.tolist(),
            "node_labels": labels,
            "deterministic": self.deterministic,
            "tpm_exact": self.tpm_is_exact(),
            "n_gates": self.n_gates,
            "gates": [self._gate_payload(g, labels) for g in self.gates],
        }

    def inspect(self) -> dict:
        """Everything `render_brain` shows, plus each gate's full table: the genome itself."""
        payload = self.render_brain()
        payload["tables"] = [g.table.tolist() for g in self.gates]
        return payload


def crossover(parent_a: MarkovGatesAgent, parent_b: MarkovGatesAgent, rng: np.random.Generator,
              name: str, seed: int) -> MarkovGatesAgent:
    """Cut each parent's gate list at a point of its own and join the head of one to the tail of
    the other, so a child can be longer or shorter than either parent.

    Gates are taken whole, never mixed, because a gate's table means something only with its own
    wiring. The child is clipped into `[MIN_GATES, MAX_GATES]`, taking one gate of `parent_a` if
    both pieces were empty. Parents of different node layouts or gate kinds are refused, since
    their gates name nodes or tables the other cannot read.
    """
    layout_a = (parent_a.n_sensor, parent_a.n_hidden, parent_a.n_motor, parent_a.deterministic)
    layout_b = (parent_b.n_sensor, parent_b.n_hidden, parent_b.n_motor, parent_b.deterministic)
    if layout_a != layout_b:
        raise ValueError(
            f"crossover requires parents with the same nodes and gate kind; got (sensor, hidden, "
            f"motor, deterministic) {layout_a} and {layout_b}")
    cut_a = int(rng.integers(0, parent_a.n_gates + 1))
    cut_b = int(rng.integers(0, parent_b.n_gates + 1))
    gates = [g.copy() for g in parent_a.gates[:cut_a]] + [g.copy() for g in parent_b.gates[cut_b:]]
    if not gates:
        gates = [parent_a.gates[int(rng.integers(parent_a.n_gates))].copy()]
    return parent_a._with_gates(gates[:MAX_GATES], name, seed)


def spawn(name: str, game: Any, seed: int, config: Any) -> MarkovGatesAgent:
    """One random gate brain for `engine.evolution.evolve`.

    `n_hidden` is read off the config as the sigmoid brain reads it. `n_gates` and
    `deterministic` come from `config.genome`, and any other key there is refused: a knob this
    genome does not have would otherwise be accepted, ignored, and recorded as the setting the run
    used. Matches `engine.evolution.SpawnFn`.
    """
    knobs = dict(getattr(config, "genome", None) or {})
    unknown = sorted(set(knobs) - set(GENOME_KNOBS))
    if unknown:
        raise ValueError(f"a gate brain's genome takes {', '.join(GENOME_KNOBS)}, not "
                         f"{', '.join(unknown)}")
    return MarkovGatesAgent.random(
        name, game.n_states, game.n_actions,
        n_hidden=int(getattr(config, "n_hidden", 2)),
        n_gates=int(knobs.get("n_gates", DEFAULT_N_GATES)),
        deterministic=parse_deterministic(knobs.get("deterministic", True)),
        seed=seed, start_state=start_observation(game),
    )
