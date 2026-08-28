"""safeflow -- the excess-flow safety criterion in closed matrix form.

Reference implementation for REPORT.md.  Single file, NumPy only.

The criterion of Khan & Tomescu (arXiv:2102.06480v1) reduces to one nonnegative
edge vector and a dot product:

    lambda = (S_out - I) f          S_out = T^T T, the sibling matrix
    f_P    = f_out(u_1) - lambda^T x_P

Everything here follows from that identity.  Section numbers in docstrings
refer to REPORT.md.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FlowDAG", "ConservationError", "incidence", "excess", "excess_dual",
    "excess_definition", "is_safe", "is_left_maximal", "is_right_maximal",
    "classify", "closure_from", "closure_all", "leak_matrix", "rollout",
    "maximal_safe_paths", "all_paths", "figure1", "uniform_regular_dag",
]


class ConservationError(ValueError):
    """Raised when a flow violates f_in(v) = f_out(v) at an internal vertex."""


# --------------------------------------------------------------------------- #
# Section 1 -- the graph and its incidence algebra
# --------------------------------------------------------------------------- #

@dataclass
class FlowDAG:
    """A DAG with a conserved flow.  Parallel edges are permitted.

    src[e], dst[e] are the tail and head of edge e; f[e] its flow.  These are
    the nonzero patterns of the unsigned incidence matrices T and H, held as
    index arrays rather than materialised.
    """

    src: np.ndarray
    dst: np.ndarray
    f: np.ndarray
    n: int

    f_out: np.ndarray = field(init=False)
    f_in: np.ndarray = field(init=False)
    lam: np.ndarray = field(init=False)          # (S_out - I) f      -- eq. (4)
    mu: np.ndarray = field(init=False)           # (S_in  - I) f      -- eq. (5)
    lam_min_out: np.ndarray = field(init=False)  # eq. (6)
    mu_min_in: np.ndarray = field(init=False)
    order: np.ndarray = field(init=False)        # a topological vertex order
    rank: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.f = np.asarray(self.f, dtype=np.float64)
        if not len(self.src) == len(self.dst) == len(self.f):
            raise ValueError("src, dst and f must have equal length")

        self.f_out = np.bincount(self.src, weights=self.f, minlength=self.n)
        self.f_in = np.bincount(self.dst, weights=self.f, minlength=self.n)

        # The leak vectors.  lambda_e is the total flow on e's out-siblings and
        # mu_e the total flow on its in-siblings -- Proposition 2.
        self.lam = self.f_out[self.src] - self.f
        self.mu = self.f_in[self.dst] - self.f

        self.lam_min_out = _segment_min(self.lam, self.src, self.n)
        self.mu_min_in = _segment_min(self.mu, self.dst, self.n)

        self.order = _topological_order(self.src, self.dst, self.n)
        self.rank = np.empty(self.n, dtype=np.int64)
        self.rank[self.order] = np.arange(self.n)

    @property
    def m(self) -> int:
        return len(self.f)

    def out_edges(self, v: int) -> np.ndarray:
        return np.flatnonzero(self.src == v)

    def in_edges(self, v: int) -> np.ndarray:
        return np.flatnonzero(self.dst == v)

    def check_conservation(self, atol: float = 1e-9) -> None:
        """Assert (H - T) f = 0 on internal vertices.  See REPORT.md section 3.

        Theorem 1 does not need this; Corollary 1 and hence the minimax
        guarantee of Theorem A do.
        """
        internal = np.flatnonzero((self.f_in > 0) & (self.f_out > 0))
        bad = np.abs(self.f_in[internal] - self.f_out[internal]) > atol
        if bad.any():
            v = int(internal[np.argmax(np.abs(self.f_in[internal] - self.f_out[internal]))])
            raise ConservationError(
                f"{int(bad.sum())} internal vertices violate conservation; worst is "
                f"{v} with |f_in - f_out| = {abs(self.f_in[v] - self.f_out[v]):.3e}")

    def edge_path(self, nodes) -> np.ndarray:
        """Cheapest edge sequence realising a vertex walk (min leak per hop)."""
        out = []
        for a, b in zip(nodes, nodes[1:]):
            cand = np.flatnonzero((self.src == a) & (self.dst == b))
            if not cand.size:
                raise ValueError(f"no edge {a} -> {b}")
            out.append(int(cand[np.argmin(self.lam[cand])]))
        return np.array(out, dtype=np.int64)


def _topological_order(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    """Kahn's algorithm.  Vertex ids are not assumed to be topological -- in the
    reference instance of `figure1` they are not."""
    indeg = np.bincount(dst, minlength=n).astype(np.int64)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
    stack = [int(v) for v in np.flatnonzero(indeg == 0)]
    order: list[int] = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != n:
        raise ValueError("graph is not acyclic")
    return np.array(order, dtype=np.int64)


def _segment_min(values: np.ndarray, keys: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.inf)
    if values.size:
        perm = np.argsort(keys, kind="stable")
        k = keys[perm]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        out[k[starts]] = np.minimum.reduceat(values[perm], starts)
    return out


def incidence(dag: FlowDAG):
    """The dense T and H of equation (1).  For tests and small examples only."""
    T = np.zeros((dag.n, dag.m))
    H = np.zeros((dag.n, dag.m))
    T[dag.src, np.arange(dag.m)] = 1.0
    H[dag.dst, np.arange(dag.m)] = 1.0
    return T, H


# --------------------------------------------------------------------------- #
# Sections 2-4 -- excess flow, safety, maximality
# --------------------------------------------------------------------------- #

def excess(dag: FlowDAG, path) -> float:
    """Theorem 1:  f_P = f_out(u_1) - lambda^T x_P.   O(|P|)."""
    e = np.asarray(path, dtype=np.int64)
    if e.size == 0:
        raise ValueError("the empty path has no excess flow")
    return float(dag.f_out[dag.src[e[0]]] - dag.lam[e].sum())


def excess_dual(dag: FlowDAG, path) -> float:
    """Corollary 1:  f_P = f_in(u_k) - mu^T x_P.  Needs conservation."""
    e = np.asarray(path, dtype=np.int64)
    return float(dag.f_in[dag.dst[e[-1]]] - dag.mu[e].sum())


def excess_definition(dag: FlowDAG, path) -> float:
    """Equation (2) literally -- an independent oracle for the tests."""
    e = np.asarray(path, dtype=np.int64)
    return float(dag.f[e].sum() - dag.f_out[dag.dst[e[:-1]]].sum())


def is_safe(dag: FlowDAG, path, eps: float = 0.0) -> bool:
    return excess(dag, path) > eps


def is_right_maximal(dag: FlowDAG, path, eps: float = 0.0) -> bool:
    """Proposition 3:  f_P <= lambda_min_out(u_k)."""
    e = np.asarray(path, dtype=np.int64)
    step = dag.lam_min_out[dag.dst[e[-1]]]
    return True if not np.isfinite(step) else excess(dag, e) - step <= eps


def is_left_maximal(dag: FlowDAG, path, eps: float = 0.0) -> bool:
    """Proposition 3:  f_P <= mu_min_in(u_1)."""
    e = np.asarray(path, dtype=np.int64)
    step = dag.mu_min_in[dag.src[e[0]]]
    return True if not np.isfinite(step) else excess(dag, e) - step <= eps


def classify(dag: FlowDAG, path, eps: float = 0.0) -> str:
    if not is_safe(dag, path, eps):
        return "unsafe"
    if is_left_maximal(dag, path, eps) and is_right_maximal(dag, path, eps):
        return "maximal safe"
    return "safe, extends"


# --------------------------------------------------------------------------- #
# Section 6 -- the tropical closure
# --------------------------------------------------------------------------- #

def leak_matrix(dag: FlowDAG) -> np.ndarray:
    """L of equation (8): min leak per vertex pair, zero on the diagonal."""
    L = np.full((dag.n, dag.n), np.inf)
    np.minimum.at(L, (dag.src, dag.dst), dag.lam)
    np.fill_diagonal(L, 0.0)
    return L


def closure_from(dag: FlowDAG, source: int) -> np.ndarray:
    """Max excess flow over paths starting at `source`.  One O(m) sweep.

    Edges are relaxed in topological order of their tail, so dist[u] is final
    before u's out-edges are read.
    """
    order = np.argsort(dag.rank[dag.src], kind="stable")
    dist = np.full(dag.n, np.inf)
    dist[source] = 0.0
    for e in order:
        cand = dist[dag.src[e]] + dag.lam[e]
        if cand < dist[dag.dst[e]]:
            dist[dag.dst[e]] = cand
    out = dag.f_out[source] - dist
    out[source] = -np.inf
    return out


def closure_all(dag: FlowDAG) -> np.ndarray:
    """F = f_out 1^T - L*  of equation (9)."""
    F = np.stack([closure_from(dag, s) for s in range(dag.n)])
    return F


# --------------------------------------------------------------------------- #
# Section 7 -- the sum-product reading
# --------------------------------------------------------------------------- #

def rollout(dag: FlowDAG, path) -> float:
    """R(P) = f(e_1) * prod M_e, the proportional-split mass.  f_P <= R(P)."""
    e = np.asarray(path, dtype=np.int64)
    denom = dag.f_out[dag.src]
    M = np.divide(dag.f, denom, out=np.zeros_like(dag.f), where=denom > 0)
    return float(dag.f[e[0]] * np.prod(M[e[1:]]))


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #

def all_paths(dag: FlowDAG, max_len: int = 10):
    """Every edge sequence, up to max_len edges.  Exponential; tests only."""
    out = []
    for e0 in range(dag.m):
        stack = [[e0]]
        while stack:
            p = stack.pop()
            out.append(list(p))
            if len(p) < max_len:
                for e in dag.out_edges(int(dag.dst[p[-1]])):
                    stack.append(p + [int(e)])
    return out


def maximal_safe_paths(dag: FlowDAG, eps: float = 0.0) -> list[tuple[tuple[int, ...], float]]:
    """All maximal safe paths, by exhaustive search plus the tests of (7).

    Deliberately simple: the point of this file is the criterion, not the
    enumeration algorithm.  For the sliding-window method (and the erratum
    discussed in REPORT.md section 5) see the discussion there.
    """
    out = []
    for p in all_paths(dag):
        if (is_safe(dag, p, eps) and is_left_maximal(dag, p, eps)
                and is_right_maximal(dag, p, eps)):
            out.append((tuple(p), excess(dag, p)))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #

_F1_V = ["v0", "v1", "v2", "v3", "v4", "v5", "s_a", "s_b",
         "j1", "j2", "j3", "j4", "t1", "t2", "t3", "t4", "t5", "t6"]
_F1_E = [("s_a", "v0", 3.), ("s_b", "v0", 4.),
         ("v0", "v1", 7.), ("v1", "v2", 9.), ("v2", "v3", 8.),
         ("v3", "v4", 6.), ("v4", "v5", 5.),
         ("j1", "v1", 4.), ("j2", "v2", 2.), ("j3", "v3", 3.), ("j4", "v4", 1.),
         ("v1", "t1", 2.), ("v2", "t2", 3.), ("v3", "t3", 5.), ("v4", "t4", 2.),
         ("v5", "t5", 2.), ("v5", "t6", 3.)]


def figure1():
    """Khan & Tomescu's motivating instance, with source/sink stubs added so
    that conservation holds exactly at all six chain vertices."""
    ix = {nm: i for i, nm in enumerate(_F1_V)}
    dag = FlowDAG(src=np.array([ix[a] for a, _, _ in _F1_E]),
                  dst=np.array([ix[b] for _, b, _ in _F1_E]),
                  f=np.array([w for _, _, w in _F1_E]), n=len(_F1_V))
    dag.check_conservation()
    return dag, ix


def uniform_regular_dag(layers: int, width: int, d: int, throughput: float = 1.0):
    """A layered DAG in which every non-sink vertex has out-degree d and the
    flow splits uniformly -- the hypothesis of Proposition 7 (section 10).

    Vertex (i, j) sends to (i+1, (j+t) mod width) for t = 0..d-1, so internal
    vertices have in-degree d as well and every vertex carries `throughput`.
    """
    if d > width:
        raise ValueError("d must not exceed the layer width")
    idx = lambda i, j: i * width + j
    src, dst, f = [], [], []
    for i in range(layers - 1):
        for j in range(width):
            for t in range(d):
                src.append(idx(i, j))
                dst.append(idx(i + 1, (j + t) % width))
                f.append(throughput / d)
    dag = FlowDAG(src=np.array(src), dst=np.array(dst), f=np.array(f),
                  n=layers * width)
    dag.check_conservation()
    return dag
