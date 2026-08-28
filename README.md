# safe-flow-matrix-formulation

A closed matrix form for the excess-flow safety criterion of Khan & Tomescu
([arXiv:2102.06480v1](https://arxiv.org/abs/2102.06480v1)), and what it implies
about which feedforward computational graphs the framework can speak of.

```
λ  = (Sₒᵤₜ − I) f        Sₒᵤₜ = TᵀT, the sibling matrix
f_P = f_out(u₁) − λᵀx_P
```

- **[REPORT.md](REPORT.md)** — the derivation: Propositions 1–7, Corollaries 1–3, with proofs.
- **[safeflow.py](safeflow.py)** — minimal reference implementation. NumPy only, single file.
- **[test_safeflow.py](test_safeflow.py)** — 16 tests, one per result, checked against
  exhaustive path enumeration where feasible.

```bash
python test_safeflow.py          # or: pytest test_safeflow.py
```

```python
import safeflow as sf

dag, ix = sf.figure1()                                   # the worked instance
p = dag.edge_path([ix["v0"], ix["v1"], ix["v2"], ix["v3"]])
sf.excess(dag, p), sf.classify(dag, p)                   # (2.0, 'maximal safe')

g = sf.uniform_regular_dag(layers=5, width=6, d=3)       # Proposition 7
{len(p) for p, _ in sf.maximal_safe_paths(g)}            # {1} -- only single edges
```

The headline result of §10: on a DAG where every non-sink vertex has out-degree
`d ≥ 2` and the flow splits uniformly, **no path of two or more edges is safe** —
under precisely the hypothesis of Proposition 4.2 of Vitvitskyi et al.,
[*What Makes a Good Feedforward Computational Graph?*](https://arxiv.org/abs/2502.06751)
The better the computational graph by that paper's mixing criterion, the less any
decomposition-invariant method can say about it.
