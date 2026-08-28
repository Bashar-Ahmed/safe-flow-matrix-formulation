"""Numerical verification of every proposition in REPORT.md.

Run with `pytest test_safeflow.py` or directly with `python test_safeflow.py`.
Each test name states the result it checks.
"""
from __future__ import annotations

import itertools

import numpy as np

import safeflow as sf

CHAIN = ["v0", "v1", "v2", "v3", "v4", "v5"]
CLOSE = lambda a, b: abs(a - b) < 1e-9


def chain_windows(dag, ix):
    for i, j in itertools.combinations(range(len(CHAIN)), 2):
        yield (CHAIN[i], CHAIN[j]), dag.edge_path([ix[c] for c in CHAIN[i:j + 1]])


# --- Section 2 ------------------------------------------------------------- #

def test_proposition_1_internal_vertex_identity():
    """y_P = T x_P - delta_{u_1} = H x_P - delta_{u_k}."""
    dag, ix = sf.figure1()
    T, H = sf.incidence(dag)
    for _k, path in chain_windows(dag, ix):
        x = np.zeros(dag.m); x[path] = 1.0
        y = np.zeros(dag.n)
        y[dag.dst[path[:-1]]] = 1.0                     # internal vertices
        d1 = np.zeros(dag.n); d1[dag.src[path[0]]] = 1.0
        dk = np.zeros(dag.n); dk[dag.dst[path[-1]]] = 1.0
        assert np.allclose(y, T @ x - d1)
        assert np.allclose(y, H @ x - dk)


def test_theorem_1_identity_and_nonnegativity():
    """f_P = f_out(u_1) - lambda^T x_P, with lambda = (T^T T - I) f >= 0."""
    dag, ix = sf.figure1()
    T, _H = sf.incidence(dag)
    lam_dense = (T.T @ T - np.eye(dag.m)) @ dag.f
    assert np.allclose(lam_dense, dag.lam)
    assert (dag.lam >= -1e-12).all()
    for _k, path in chain_windows(dag, ix):
        assert CLOSE(sf.excess(dag, path), sf.excess_definition(dag, path))


def test_theorem_1_matches_published_values():
    """The excess flows printed in the source paper's worked instance."""
    want = {("v0", "v1"): 7, ("v0", "v2"): 5, ("v0", "v3"): 2, ("v0", "v4"): -3,
            ("v0", "v5"): -5, ("v1", "v2"): 9, ("v1", "v3"): 6, ("v1", "v4"): 1,
            ("v1", "v5"): -1, ("v2", "v3"): 8, ("v2", "v4"): 3, ("v2", "v5"): 1,
            ("v3", "v4"): 6, ("v3", "v5"): 4, ("v4", "v5"): 5}
    dag, ix = sf.figure1()
    for key, path in chain_windows(dag, ix):
        assert CLOSE(sf.excess(dag, path), want[key]), key


# --- Section 3 ------------------------------------------------------------- #

def test_proposition_2_lambda_is_the_sibling_sum():
    """lambda_e = total flow on e's out-siblings; mu_e on its in-siblings."""
    dag, _ix = sf.figure1()
    for e in range(dag.m):
        out_sibs = [g for g in range(dag.m) if dag.src[g] == dag.src[e] and g != e]
        in_sibs = [g for g in range(dag.m) if dag.dst[g] == dag.dst[e] and g != e]
        assert CLOSE(dag.lam[e], dag.f[out_sibs].sum())
        assert CLOSE(dag.mu[e], dag.f[in_sibs].sum())


def test_corollary_1_dual_form():
    """f_P = f_in(u_k) - mu^T x_P, on a conserved flow."""
    dag, ix = sf.figure1()
    for _k, path in chain_windows(dag, ix):
        assert CLOSE(sf.excess_dual(dag, path), sf.excess(dag, path))


def test_theorem_1_survives_without_conservation_but_corollary_1_does_not():
    """Conservation is needed by Corollary 1 only -- REPORT.md section 3.

    The two forms differ by the sum of (f_out - f_in) over the *internal*
    vertices of P, so a violation created on a path edge joining two of them
    contributes +d at its tail and -d at its head and cancels exactly.  The
    perturbation must therefore touch a vertex the path passes through via an
    edge the path does not use -- here the leak edge v1 -> t1.
    """
    dag, ix = sf.figure1()
    leak = int(dag.edge_path([ix["v1"], ix["t1"]])[0])
    f = dag.f.copy(); f[leak] += 1.4                      # break conservation at v1
    broken = sf.FlowDAG(src=dag.src, dst=dag.dst, f=f, n=dag.n)
    path = broken.edge_path([ix["v0"], ix["v1"], ix["v2"], ix["v3"]])
    assert CLOSE(sf.excess(broken, path), sf.excess_definition(broken, path))
    assert CLOSE(sf.excess(broken, path), 2.0 - 1.4)      # diverging form shifts
    assert CLOSE(sf.excess_dual(broken, path), 2.0)       # converging form does not
    assert not CLOSE(sf.excess_dual(broken, path), sf.excess(broken, path))
    try:
        broken.check_conservation(); raise AssertionError("should have raised")
    except sf.ConservationError:
        pass


def test_corollary_2_subpath_closure():
    """Trimming either end cannot decrease the excess flow."""
    dag, ix = sf.figure1()
    full = dag.edge_path([ix[c] for c in CHAIN])
    base = sf.excess(dag, full)
    for i in range(len(full)):
        for j in range(i, len(full)):
            assert sf.excess(dag, full[i:j + 1]) >= base - 1e-9


def test_corollary_3_incremental_updates():
    """Appending e costs lambda_e; prepending it costs mu_e."""
    dag, ix = sf.figure1()
    p = dag.edge_path([ix["v1"], ix["v2"], ix["v3"]])
    right = dag.edge_path([ix["v3"], ix["v4"]])
    left = dag.edge_path([ix["v0"], ix["v1"]])
    assert CLOSE(sf.excess(dag, np.r_[p, right]), sf.excess(dag, p) - dag.lam[right[0]])
    assert CLOSE(sf.excess(dag, np.r_[left, p]), sf.excess(dag, p) - dag.mu[left[0]])


# --- Section 4 ------------------------------------------------------------- #

def test_proposition_3_maximality_against_brute_force():
    """The two O(1) threshold tests agree with explicit extension search."""
    dag, _ix = sf.figure1()
    safe = {tuple(p) for p in sf.all_paths(dag, max_len=6) if sf.is_safe(dag, p)}
    for p in safe:
        end, start = int(dag.dst[p[-1]]), int(dag.src[p[0]])
        brute_r = not any(tuple(list(p) + [int(e)]) in safe for e in dag.out_edges(end))
        brute_l = not any(tuple([int(e)] + list(p)) in safe for e in dag.in_edges(start))
        assert sf.is_right_maximal(dag, p) == brute_r, p
        assert sf.is_left_maximal(dag, p) == brute_l, p


def test_the_three_chain_maximal_paths():
    """The paper's worked instance has exactly these on the chain."""
    dag, ix = sf.figure1()
    found = dict(sf.maximal_safe_paths(dag))
    for (a, b), want in {("v0", "v3"): 2., ("v1", "v4"): 1., ("v2", "v5"): 1.}.items():
        lo, hi = int(a[1]), int(b[1])
        key = tuple(int(e) for e in dag.edge_path([ix[f"v{k}"] for k in range(lo, hi + 1)]))
        assert key in found and CLOSE(found[key], want), (a, b)


def test_e1_a_window_can_fail_locally_yet_extend_globally():
    """A window unextendable along one out-edge may extend via a heavier
    sibling -- the defect discussed in REPORT.md section 5."""
    names = ["x", "j", "y", "k", "u", "p", "q", "t"]
    ix = {nm: i for i, nm in enumerate(names)}
    E = [("x", "y", 4.), ("j", "y", 2.), ("y", "u", 5.), ("y", "t", 1.),
         ("k", "u", 1.), ("u", "p", 5.), ("u", "q", 1.)]
    dag = sf.FlowDAG(src=np.array([ix[a] for a, _, _ in E]),
                     dst=np.array([ix[b] for _, b, _ in E]),
                     f=np.array([w for _, _, w in E]), n=len(names))
    dag.check_conservation()
    w = dag.edge_path([ix["x"], ix["y"], ix["u"]])
    assert CLOSE(sf.excess(dag, w), 3.0)
    via_q = np.r_[w, dag.edge_path([ix["u"], ix["q"]])]
    via_p = np.r_[w, dag.edge_path([ix["u"], ix["p"]])]
    assert sf.excess(dag, via_q) < 0 and sf.excess(dag, via_p) > 0
    assert not sf.is_right_maximal(dag, w)      # the O(1) test catches it


# --- Section 6 ------------------------------------------------------------- #

def test_proposition_4_closure_matches_brute_force_and_the_naive_power():
    """F = f_out 1^T - L*, three ways."""
    dag, _ix = sf.figure1()
    F = sf.closure_all(dag)
    best = np.full((dag.n, dag.n), -np.inf)
    for p in sf.all_paths(dag, max_len=8):
        s, t = int(dag.src[p[0]]), int(dag.dst[p[-1]])
        best[s, t] = max(best[s, t], sf.excess(dag, p))
    L = sf.leak_matrix(dag)
    P = L.copy()
    for _ in range(dag.n - 1):
        P = np.min(P[:, :, None] + L[None, :, :], axis=1)
    naive = dag.f_out[:, None] - P
    np.fill_diagonal(naive, -np.inf)
    for s, t in itertools.permutations(range(dag.n), 2):
        if np.isneginf(best[s, t]) and np.isneginf(F[s, t]):
            continue
        assert CLOSE(F[s, t], best[s, t]), (s, t)
        assert CLOSE(F[s, t], naive[s, t]), (s, t)


# --- Section 7 ------------------------------------------------------------- #

def test_proposition_5_excess_never_exceeds_rollout():
    """f_P <= R(P), with equality on single edges."""
    dag, _ix = sf.figure1()
    for e in range(dag.m):
        assert CLOSE(sf.excess(dag, [e]), sf.rollout(dag, [e]))
    for p in sf.all_paths(dag, max_len=6):
        assert sf.excess(dag, p) <= sf.rollout(dag, p) + 1e-9
    g = sf.uniform_regular_dag(4, 5, 2)
    for p in sf.all_paths(g, max_len=4):
        assert sf.excess(g, p) <= sf.rollout(g, p) + 1e-9


# --- Section 10 ------------------------------------------------------------ #

def test_proposition_6_leaks_are_complements_of_W_and_Delta():
    """lambda_e = f_out(tail)(1 - W_e) and mu_e = f_in(head)(1 - Delta_e);
    and on a uniform graph W_e = 1/outdeg, Delta_e = 1/indeg exactly."""
    for dag in (sf.figure1()[0], sf.uniform_regular_dag(4, 6, 3)):
        W = dag.f / dag.f_out[dag.src]
        D = dag.f / dag.f_in[dag.dst]
        assert np.allclose(dag.lam, dag.f_out[dag.src] * (1 - W))
        assert np.allclose(dag.mu, dag.f_in[dag.dst] * (1 - D))

    g = sf.uniform_regular_dag(4, 6, 3)
    outdeg = np.bincount(g.src, minlength=g.n)
    indeg = np.bincount(g.dst, minlength=g.n)
    W = g.f / g.f_out[g.src]
    D = g.f / g.f_in[g.dst]
    assert np.allclose(W, 1.0 / outdeg[g.src])            # Vitvitskyi et al. W
    assert np.allclose(D, 1.0 / indeg[g.dst])             # Vitvitskyi et al. Delta


def test_proposition_7_uniform_collapse():
    """On a d-regular uniformly-splitting DAG with d >= 2, only single edges
    are safe, and f_P = Phi (1 - l (1 - 1/d)) exactly."""
    for d in (2, 3, 4):
        g = sf.uniform_regular_dag(layers=5, width=6, d=d, throughput=1.0)
        for p in sf.all_paths(g, max_len=4):
            l = len(p)
            predicted = 1.0 * (1 - l * (1 - 1 / d))
            assert CLOSE(sf.excess(g, p), predicted), (d, l)
            assert sf.is_safe(g, p) == (l == 1), (d, l)
        assert all(len(p) == 1 for p, _x in sf.maximal_safe_paths(g))


def test_proposition_7_boundary_out_degree_one_permits_long_paths():
    """The bound l - 1 < 1/(d-1) is tight: at d = 1 arbitrarily long paths
    are safe, which is why the collapse is a statement about branching."""
    g = sf.uniform_regular_dag(layers=6, width=4, d=1, throughput=1.0)
    longest = max(sf.all_paths(g, max_len=6), key=len)
    assert len(longest) == 5
    assert sf.is_safe(g, longest) and CLOSE(sf.excess(g, longest), 1.0)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} tests passed.")
