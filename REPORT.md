# Matrix Formulation of the Safe Flow Framework

**A closed form for the excess-flow safety criterion, and what it says about which computational graphs the framework can speak of.**

---

## Contents

1. [Summary](#1-summary)
2. [Setting](#2-setting)
3. [The reduction](#3-the-reduction)
4. [Immediate consequences](#4-immediate-consequences)
5. [Safety and maximality are linear thresholds](#5-safety-and-maximality-are-linear-thresholds)
6. [Bilinearity and relaxation](#6-bilinearity-and-relaxation)
7. [The all-pairs table is a tropical closure](#7-the-all-pairs-table-is-a-tropical-closure)
8. [Two semirings on one DAG](#8-two-semirings-on-one-dag)
9. [Correspondence with the source paper](#9-correspondence-with-the-source-paper)
10. [Which graphs the framework can speak about](#10-which-graphs-the-framework-can-speak-about)
11. [Scope and preconditions](#11-scope-and-preconditions)
12. [Verification](#12-verification)
13. [References](#13-references)

---

## 1. Summary

Khan and Tomescu [1] characterise *safe* paths in a flow DAG — those appearing in every flow decomposition — by a quantity they call excess flow, defined by a summation over the path's internal vertices. Their presentation is combinatorial throughout.

The quantity has a closed form. Writing $T$ and $H$ for the unsigned tail and head incidence matrices and $f$ for the flow,

$$\boxed{f_P = f_{\mathrm{out}}(u_1)-\lambda^\top x_P,\qquad \lambda = (T^\top T - I)\,f \ge 0}$$

where $x_P$ is the path's edge-indicator vector. A budget granted at the first vertex, minus a fixed linear cost accumulated over the edges traversed. The rest of this report is consequences.

| § | Result |
|---|---|
| 3 | **Theorem 1.** The identity above; the internal-vertex set is not independent data. |
| 4 | **Proposition 2.** $T^\top T$ is the paper's own *sibling* relation, block diagonal; $\lambda_e$ is the flow on $e$'s out-siblings. Corollaries 1–3 recover Lemmas 2(a) and 2(c) as one-line facts. |
| 5 | **Proposition 3.** Safety and both maximality conditions are three linear thresholds. |
| 6 | $f_P$ is bilinear in (flow, path), so $\partial f_P/\partial f$ is closed-form; the LP relaxation has integral optima. |
| 7 | **Proposition 4.** The all-pairs table is $F = f_{\mathrm{out}}\mathbf{1}^\top \ominus L^*$, a tropical matrix closure. |
| 8 | **Proposition 5.** $f_P \le R(P)$: excess flow is a certified floor beneath the proportional-split path mass. |
| 10 | **Propositions 6 and 7.** $\lambda$ and $\mu$ are the unnormalised complements of the walk and diffusion matrices of Vitvitskyi et al. [2]. On a $d$-regular uniformly-splitting DAG with $d \ge 2$, *no path of two or more edges is safe* — under precisely the hypothesis of their Proposition 4.2. |

Every proposition is checked numerically by `test_safeflow.py` (§12), including against exhaustive path enumeration.

---

## 2. Setting

Let $G = (V,E)$ be a directed acyclic multigraph, $n = |V|$, $m = |E|$, carrying a flow $f \in \mathbb{R}^m_{>0}$. Parallel edges are permitted and are the normal case in transformer circuit graphs, where a $q/k/v$ triple joins the same pair of components.

Define the two **unsigned incidence matrices** $T, H \in \{0,1\}^{n\times m}$:

$$
\begin{array}{rl}
\displaystyle T_{v,e} = 1 \iff v = \mathrm{tail}(e), \qquad H_{v,e} = 1 \iff v = \mathrm{head}(e). & \qquad (1)
\end{array}
$$

Then $f_{\mathrm{out}} = Tf$, $f_{\mathrm{in}} = Hf$, and flow conservation is the single linear constraint $(H-T)f = 0$ on internal vertices. A path $P = (u_1,\dots,u_k)$ with edges $e_1,\dots,e_{k-1}$ is carried by its **edge-indicator vector** $x_P \in \{0,1\}^m$; $\delta_v$ denotes a vertex one-hot.

### 2.1 Safety is a minimax property

A decomposition $\mathcal{D}$ is a set of weighted source-to-sink paths whose per-edge weight sums reproduce $f$. Decompositions are not unique, and the framework's premise is to report only what survives the choice. Writing $W(P,\mathcal{D})$ for the total weight of paths in $\mathcal{D}$ containing $P$ as a subpath:

$$
\begin{array}{rl}
\displaystyle P \text{ is } w\text{-safe} \iff \min_{\mathcal{D}} W(P,\mathcal{D}) \ge w > 0. & \qquad (2)
\end{array}
$$

The guarantee is against an adversary who chooses the decomposition after seeing the path. Nothing is claimed about a typical decomposition — a distinction that becomes the whole content of §8.

### 2.2 Excess flow

**Definition** (Khan–Tomescu, Definition 1 / Lemma 2b). For a path $P$,

$$
\begin{array}{rl}
\displaystyle f_P = f^\top x_P - f_{\mathrm{out}}^\top y_P, & \qquad (3)
\end{array}
$$

where $y_P \in \{0,1\}^n$ indicates the **internal** vertices of $P$ — every vertex except the two endpoints.

**Theorem A** (Khan–Tomescu, Theorem 1). $P$ is $w$-safe iff $f_P \ge w > 0$; in particular $f_P$ attains the minimum in (2). *Taken as given throughout.*

### 2.3 The obstacle

Equation (3) appears to require two independent descriptions of a path: an edge set $x_P$ and a vertex set $y_P$. Were that so, excess flow would not be a linear functional of the path, and the incremental structure the source paper exploits would need re-establishing for each of its algorithms separately.

---

## 3. The reduction

**Proposition 1.** For any path $P$ from $u_1$ to $u_k$,

$$
\begin{array}{rl}
\displaystyle y_P = T x_P - \delta_{u_1} = H x_P - \delta_{u_k}. & \qquad (4)
\end{array}
$$

*Proof.* $(Tx_P)_v$ counts the edges of $P$ leaving $v$. A path leaves each of $u_1,\dots,u_{k-1}$ exactly once and leaves $u_k$ never, so $Tx_P$ is the indicator of $\{u_1,\dots,u_{k-1}\}$; removing $\delta_{u_1}$ leaves the internal vertices. The second identity is the mirror argument on $H$, whose indicator is $\{u_2,\dots,u_k\}$. $\square$

**Theorem 1.** Let $S_{\mathrm{out}} = T^\top T$ and $\lambda = (S_{\mathrm{out}} - I)f$. Then $\lambda \ge 0$ componentwise, $\lambda$ depends only on the flow, and for every path

$$
\begin{array}{rl}
\displaystyle f_P = f_{\mathrm{out}}(u_1) - \lambda^\top x_P. & \qquad (5)
\end{array}
$$

*Proof.* Substituting the left identity of (4) into (3),

$$
\begin{aligned}
f_P &= f^\top x_P - (Tf)^\top\!\left(Tx_P - \delta_{u_1}\right)\\
    &= f^\top x_P - f^\top T^\top T x_P + (Tf)^\top\delta_{u_1}\\
    &= f_{\mathrm{out}}(u_1) - \big[(S_{\mathrm{out}}-I)f\big]^\top x_P .
\end{aligned}
$$

For nonnegativity, $(S_{\mathrm{out}}f)_e = \sum_{g:\ \mathrm{tail}(g)=\mathrm{tail}(e)} f(g) = f_{\mathrm{out}}(\mathrm{tail}\,e) \ge f(e)$, since $e$ is itself one of the summands and $f \ge 0$. Hence $\lambda_e = f_{\mathrm{out}}(\mathrm{tail}\,e) - f(e) \ge 0$. $\square$

**Reading.** A budget of $f_{\mathrm{out}}(u_1)$ is granted at the start vertex; each edge traversed charges its **leak** $\lambda_e$ — the flow that may depart at that edge's tail instead of continuing; the path is safe for as long as the budget stays positive. Since $\lambda \ge 0$ the budget is a monotone staircase, so the safe region is a prefix and any window's excess is a difference of two heights. Prefix-sum structure, $O(1)$ endpoint updates and sliding-window enumeration are all readings of that one fact.

---

## 4. Immediate consequences

**Proposition 2 (the sibling matrix).** $(S_{\mathrm{out}})_{ef} = 1$ exactly when $e$ and $f$ share a tail vertex — the *siblings* of Khan–Tomescu §4. Ordering edges by tail makes $S_{\mathrm{out}}$ block diagonal, one all-ones block per vertex of size $\deg^+(v)$. Consequently

$$
\begin{array}{rl}
\displaystyle \lambda_e = \sum_{g \ne e,\ \mathrm{tail}(g)=\mathrm{tail}(e)} f(g), & \qquad (6)
\end{array}
$$

the total flow on $e$'s out-siblings.

*Proof.* $(T^\top T)_{ef} = \sum_v T_{v,e}T_{v,f} = 1$ iff some vertex is the tail of both. Subtracting $I$ deletes the diagonal. $\square$

The $m \times m$ object is a proof device. The quantity it names is a grouped sum over the edge list, computable in $O(m)$ time and $O(m)$ space; no implementation materialises $S_{\mathrm{out}}$.

**Corollary 1 (the dual).** With $\mu = (S_{\mathrm{in}} - I)f$ and $S_{\mathrm{in}} = H^\top H$,

$$
\begin{array}{rl}
\displaystyle f_P = f_{\mathrm{in}}(u_k) - \mu^\top x_P. & \qquad (7)
\end{array}
$$

*Proof.* As Theorem 1, via the right identity of (4) together with $f_{\mathrm{out}}^\top y_P = f_{\mathrm{in}}^\top y_P$, which holds because $y_P$ selects internal vertices only, where conservation gives $f_{\mathrm{in}} = f_{\mathrm{out}}$. $\square$

> ### Conservation enters exactly here
>
> Theorem 1 is an algebraic identity: (5) reproduces Definition (3) for **any** nonnegative edge weighting, conserved or not. It is Corollary 1 — and through it Theorem A — that requires $(H-T)f = 0$. On an unconserved weighting the arithmetic still evaluates; what is lost is the minimax guarantee that made the number mean something.
>
> The two forms differ by $\sum_{\text{internal}} (f_{\mathrm{out}} - f_{\mathrm{in}})$. A violation created on a path edge joining two internal vertices contributes $+\delta$ at its tail and $-\delta$ at its head and cancels exactly, so the discrepancy is invisible to that particular test — see `test_theorem_1_survives_without_conservation_but_corollary_1_does_not`.

**Corollary 2 (subpath closure).** $\lambda \ge 0$, so trimming $P$ on the right cannot decrease $f_P$; by (7) likewise on the left. Every subpath of a safe path is safe.

This is Khan–Tomescu Lemma 2(a), proved there by dropping negative terms from a summation. Here it is a sign condition on a vector — and it is exactly the monotonicity that makes a sliding-window sweep exhaustive.

**Corollary 3 (incremental updates).** Appending edge $e$ on the right changes $f_P$ by $-\lambda_e$; prepending it changes $f_P$ by $-\mu_e$.

This is Lemma 2(c), which the source paper proves separately and uses in every one of its algorithms. Here it is a single coordinate of a dot product.

---

## 5. Safety and maximality are linear thresholds

By Theorem 1 the safety predicate is a threshold on a linear functional of the path's feature vector:

$$
\begin{array}{rl}
\displaystyle P \text{ is safe} \iff \lambda^\top x_P < f_{\mathrm{out}}(u_1). & \qquad (8)
\end{array}
$$

Maximality — that $P$ cannot be lengthened at either end while remaining safe — adds two thresholds of the same kind. Khan–Tomescu Lemma 2(d) states that if any incident extension keeps a path safe then the **maximum-weight** one does. In $\lambda$-coordinates the maximum-weight out-edge is the one with the *smallest* leak, so each candidate fan collapses to a per-vertex minimum, computable in one $O(m)$ pass:

$$
\begin{array}{rl}
\displaystyle \lambda^{\min}_{\mathrm{out}}(v) = \min_{\mathrm{tail}(e)=v}\lambda_e, \qquad \mu^{\min}_{\mathrm{in}}(v) = \min_{\mathrm{head}(e)=v}\mu_e & \qquad (9)
\end{array}
$$

(taken as $+\infty$ at sinks and sources respectively, so a path ending at a sink is right-maximal by convention).

**Proposition 3.**

$$
\begin{array}{rl}
\displaystyle \text{right-maximal} \iff f_P \le \lambda^{\min}_{\mathrm{out}}(u_k), \qquad \text{left-maximal} \iff f_P \le \mu^{\min}_{\mathrm{in}}(u_1). & \qquad (10)
\end{array}
$$

*Proof.* By Corollary 3, extending right by $e$ yields excess $f_P - \lambda_e$. Some extension is safe iff $\max_e (f_P - \lambda_e) > 0$, i.e. iff $f_P > \min_e \lambda_e$. Negate; dualise for the left. $\square$

So *"is this a maximal safe path"* is a three-unit linear-threshold circuit over $x_P$ together with the two endpoint one-hots. There is nothing to fit: the weights are determined by the flow.

**Batched verification.** Stacking $p$ candidate paths as rows of an indicator matrix $X$ and gathering their start budgets into $b$ gives $\mathbf{f}_P = b - X\lambda$ — the shape of an affine layer. Where the flow itself carries an index, $\lambda$ gains that axis and this becomes a batched product.

### 5.1 A consequence for the source paper's enumeration

The sliding-window algorithm of [1] §8 sweeps each path of a candidate flow decomposition, testing right-extension only against *the next edge of that decomposition path*. By Proposition 3 a window can fail there and still extend safely through a heavier sibling elsewhere in the graph. A concrete instance: let $u$ have in-flow 6 and out-edges of weight 5 and 1, and let a window arrive at $u$ with $f_P = 3$. Extending by the weight-1 edge gives $3 - (6-1) = -2$, unsafe; by the weight-5 edge, $3 - (6-5) = 2 > 0$, safe. A decomposition path routed through the light edge therefore emits a window that is not right-maximal.

Applying (10) at both ends before emitting repairs this in constant time per window, from arrays the preprocessing already builds. Verified in `test_e1_a_window_can_fail_locally_yet_extend_globally`.

---

## 6. Bilinearity and relaxation

$\lambda = (S_{\mathrm{out}} - I)f$ is linear in $f$ with a constant matrix, so substituting back into (5) collapses both arguments into a bilinear form:

$$
\begin{array}{rl}
\displaystyle f_P = f^\top\!\left[\,T^\top\delta_{u_1} - (S_{\mathrm{out}} - I)\,x_P\,\right]. & \qquad (11)
\end{array}
$$

Linear in $f$ for a fixed path; linear in $x_P$ for a fixed flow. The bracket is therefore $\partial f_P/\partial f$ in closed form: coordinate $e$ is $+1$ if $e$ leaves $u_1$, minus the number of $e$'s out-siblings lying on $P$. Where the flow is produced by an upstream procedure, a safety margin is an explicit differentiable function of the edge weights.

**Linear-programmatic form.** Relaxing $x_P$ to a unit $s$–$t$ flow gives

$$
\begin{array}{rl}
\displaystyle \max_x f_{\mathrm{out}}(s) - \lambda^\top x \quad\text{s.t.}\quad (H-T)x = \delta_t - \delta_s, x \ge 0. & \qquad (12)
\end{array}
$$

The incidence matrix of a digraph is totally unimodular and $G$ is acyclic, so every vertex of that polytope is integral and is an $s$–$t$ path: the LP optimum *is* the maximum-excess path, with no rounding gap and no relaxation error. Replacing $\min$ by a softmin in the closure of §7 makes the construction differentiable, interpolating between the tropical certificate and a sum-product average.

---

## 7. The all-pairs table is a tropical closure

Set $L_{uv} = \min\{\lambda_e : e = (u,v)\}$, $+\infty$ if no such edge, and $L_{vv} = 0$. In the **tropical semiring** ($\oplus = \min$, $\otimes = +$) the closure

$$
\begin{array}{rl}
\displaystyle L^{*} = \bigoplus_{k \ge 0} L^{\otimes k} = (I \oplus L)^{\otimes(n-1)} & \qquad (13)
\end{array}
$$

terminates on a DAG.

**Proposition 4.** $L^*_{uv}$ is the minimum total leak over all $u \rightsquigarrow v$ paths, and

$$
\begin{array}{rl}
\displaystyle F = f_{\mathrm{out}}\mathbf{1}^\top - L^{*}, \qquad \mathrm{Safe} = [F > 0], & \qquad (14)
\end{array}
$$

where $F_{uv}$ is the maximum excess flow attainable by any path from $u$ to $v$.

*Proof.* By (5), for fixed $u$ the excess of a $u \rightsquigarrow v$ path is $f_{\mathrm{out}}(u)$ minus its total leak; that map is order-reversing, so maximising excess is minimising leak, which is what the tropical closure computes. $\square$

The dense product is the *statement*; the evaluation is a topological relaxation sweep, $O(m)$ per source. (The reference implementation's `closure_from` relaxes edges in topological order of their tail — necessary because vertex identifiers are not assumed topological, and are not in the worked instance.)

**Layered factorisation.** On a layered DAG, (13) factorises as $L_1 \otimes \cdots \otimes L_L$, structurally identical to $W_L \cdots W_1$. Edges skipping layers are absorbed by the zero diagonal, which carries a vertex forward at no cost. Same chained matrix product, different semiring.

---

## 8. Two semirings on one DAG

Normalise the flow to the row-stochastic $M_{uv} = f(u,v)/f_{\mathrm{out}}(u)$ — the flow-following walk. The product along a path,

$$
\begin{array}{rl}
\displaystyle R(P) = f(e_1)\prod_{j\ge 2} M_{e_j}, & \qquad (15)
\end{array}
$$

is the mass traversing $P$ under **proportional splitting**. Same DAG, same layered product, different semiring — and the two are ordered.

**Proposition 5.** On a conserved, strictly positive flow, $f_P \le R(P)$ for every path.

*Proof.* Proportional splitting is itself a decomposition: weight each source-to-sink path by its Markov probability under $M$; by conservation the induced per-edge weights reproduce $f$. It routes exactly $R(P)$ through $P$. If $f_P > 0$, Theorem A puts at least $f_P$ through $P$ in *every* decomposition, hence in this one. If $f_P \le 0$ the claim is trivial, since (15) is strictly positive whenever every edge of $P$ carries flow. $\square$

| | quantity | sign | failure mode |
|---|---|---|---|
| **tropical** $(\min,+)$ | guaranteed traversal under *any* decomposition | signed; negative exactly when no guarantee exists | conservative — certifies nothing for paths that are real but not forced |
| **sum-product** $(+,\times)$ | traversal under proportional splitting | strictly positive for every live path | permissive — assigns mass to everything, so only a ranking is available |

These are not competing scores. Proposition 5 says the tropical quantity is the sum-product one with the choice of decomposition quantified away. Equality holds on single edges, which have no internal vertex.

---

## 9. Correspondence with the source paper

| In Khan & Tomescu [1] | Here |
|---|---|
| Definition 1 / Lemma 2(b) | Equation (3) |
| Theorem 1 | Theorem A, assumed |
| Prefix-sum criterion | **Theorem 1** — affine, with fixed cost vector $\lambda$ |
| Lemma 2(a), subpath closure | **Corollary 2** — the sign condition $\lambda \ge 0$ |
| Lemma 2(c), incremental update | **Corollary 3** — one coordinate of a dot product |
| Lemma 2(d), max-weight sibling | **Proposition 3** — a per-vertex minimum of $\lambda$ |
| §4, siblings (prose) | **Proposition 2** — the Gram matrix $T^\top T$, block diagonal |
| Theorem 2, verification | Equation (5) |
| Lemma 2(e), no merge-then-diverge | *no matrix form* |
| Theorem 4, funnels | *no matrix form* — rests on Lemma 2(e) |

The pattern: the paper's **local** results — those about one path and its endpoints — all collapse into statements about a single vector. Its **structural** results, describing how distinct safe paths relate to one another, do not. Lemma 2(e) and the funnel theorem it supports remain combinatorial, and this reformulation makes no claim on them.

---

## 10. Which graphs the framework can speak about

Vitvitskyi, Araújo, Lackenby and Veličković [2] ask which feedforward DAGs make good computational graphs. Indexing entries as (destination, source), and writing $\delta_{i\leftarrow}$, $\delta_{j\to}$ for in- and out-degrees, they define two matrices for $(j,i) \in E$:

$$
\begin{array}{rl}
\displaystyle W_{ij} = 1/\delta_{j\to} \qquad\text{(walk matrix)}, \qquad \Delta_{ij} = 1/\delta_{i\leftarrow} \qquad\text{(diffusion matrix)}. & \qquad (16)
\end{array}
$$

$W$ is the uniform random walk leaving a vertex, normalised by **out-degree**; $\Delta$ is uniform averaging of a vertex's inputs, normalised by **in-degree**. In their words, *"while one normalises by row, the other normalises by column."* From these come **mixing time** (convergence of $W^t$ to stationarity; lower is better) and **minimax fidelity** $\min_i \max_t \Delta^t_{\tau i}$ — over source vertices $i$, the best coefficient that vertex ever attains at the sink $\tau$ (higher is better).

### 10.1 The objects coincide

**Proposition 6.** Write $W_e = f(e)/f_{\mathrm{out}}(\mathrm{tail}\,e)$ and $\Delta_e = f(e)/f_{\mathrm{in}}(\mathrm{head}\,e)$ for the flow-weighted entries, which reduce to (16) when the flow splits uniformly. Then

$$
\begin{array}{rl}
\displaystyle \lambda_e = f_{\mathrm{out}}(\mathrm{tail}\,e)\,(1 - W_e), \qquad \mu_e = f_{\mathrm{in}}(\mathrm{head}\,e)\,(1 - \Delta_e). & \qquad (17)
\end{array}
$$

*Proof.* Immediate from the definitions in Theorem 1 and Corollary 1. $\square$

The two leak vectors are the **unnormalised complements** of the two matrices: $\lambda$ is the out-degree side, $\mu$ the in-degree side. Their row/column duality is the $T^\top T$ against $H^\top H$ duality of §4. Furthermore both of their metrics are *sum-product* objects, being powers of $W$ and $\Delta$; excess flow is the min-plus reading of the same two matrices, with Proposition 5 as the bridge.

### 10.2 The disagreement, exactly

**Proposition 7 (the uniform collapse).** Let every vertex other than the sink have out-degree $d$ and let the flow split uniformly, so each such vertex carries a common throughput $\Phi$ and $f(e) = \Phi/d$. Then $\lambda_e = \Phi(1 - 1/d)$, and a path of $\ell$ edges has

$$
\begin{array}{rl}
\displaystyle f_P = \Phi\big[\,1 - \ell(1 - 1/d)\,\big], & \qquad (18)
\end{array}
$$

which is positive iff $\ell - 1 < 1/(d-1)$. For every $d \ge 2$ this forces $\ell = 1$: **no path of two or more edges is safe.**

*Proof.* $\lambda_e = f_{\mathrm{out}}(\mathrm{tail}\,e) - f(e) = \Phi - \Phi/d$. Substituting into (5) with $f_{\mathrm{out}}(u_1) = \Phi$ gives (18). Positivity requires $\ell(1-1/d) < 1$, i.e. $\ell < d/(d-1)$; and $d/(d-1) \le 2$ for all $d \ge 2$. $\square$

The hypothesis of their Proposition 4.2 — *"the out-degree of every vertex other than $\tau$ is at least 2"* — is **exactly** the hypothesis of Proposition 7. Theirs concludes that a small mixing time forces exponentially many short paths to the sink. Proposition 7 concludes, from the same premise, that the safe set degenerates to single edges. Both follow from branching, and the bound is tight: at $d = 1$ arbitrarily long paths are safe (`test_proposition_7_boundary_out_degree_one_permits_long_paths`).

Without the uniformity assumption, apply (7) at one junction: a two-edge path through $u_2$ has $f_P = f(e_2) - (\text{flow joining at } u_2)$. At every internal vertex the path must carry more than the foreign flow entering there. **Out-degree supplies the leak; in-degree supplies the foreign flow.**

### 10.3 Which way each metric pulls

Because (5) and (7) compute the same number, a safe path needs $\lambda$ *and* $\mu$ small along its length. By (17) those are separate demands, and they land on opposite sides of the two axes:

- **Mixing time — opposed.** Small $\lambda$ means each vertex's outflow is concentrated on the path edge, $W_e$ near 1: a nearly deterministic walk, which mixes *slowly*. Fast mixing spreads the rows of $W$ thin, making $\lambda$ large and the excess negative after one or two steps.
- **Fidelity — aligned.** Small $\mu$ means each vertex's inflow is concentrated on the path edge, i.e. low effective in-degree. That is also what fidelity wants: their Proposition 5.1 shows fidelity vanishes at a vertex once its in-degree exceeds one.

```
   fidelity
   (higher)
      ^
      |   ,--- the design target                  ####################
      |  o    fast mixing AND high fidelity       #  safe flow is    #
      |  ^    -- but lambda stays large, so       #  informative     #
      |   \   the safe set still collapses        #                  #
      |    \                                      #  both lambda and #
      |     \                                     #  mu are small    #
      |      '                                    #                  #
      |                                           #        o line    #
      |                                           #          graph   #
      |  o fully connected                        ####################
      |    fast mixing, fidelity at its floor
      +-------------------------------------------------------------> mixing time
                                                                       (slower)
        lambda large  <------------------------------>  lambda -> 0
```

Moving right shrinks $\lambda$; moving up shrinks $\mu$. Excess flow needs both, so it says something only in the upper right — where mixing is slow. A graph engineered to be good on both axes (their FunSearch-discovered FS graphs: $O(\mathrm{polylog}\,n)$ mixing with better-than-fully-connected fidelity) sits in the upper **left**: it fixes the $\mu$ coordinate that safety shares and keeps the $\lambda$ coordinate that safety cannot survive. The two regions are disjoint by construction, not by accident.

### 10.4 The consequence

Safe flow is a **structural** instrument, not a universal one. It reports on graphs that route deterministically and dilute little, and reports nothing but single edges on graphs that branch. Since branching is precisely what the quality metrics reward on the mixing axis, the better a feedforward computational graph is by that criterion, the less any decomposition-invariant method can say about it.

This is a statement about the graph, not about the estimator. No amount of tuning recovers what Proposition 7 forbids.

---

## 11. Scope and preconditions

- **Conservation is load-bearing but localised.** Theorem 1 holds for any nonnegative edge weighting; Corollary 1, and through it the minimax guarantee of Theorem A, needs $(H-T)f = 0$. An unconserved weighting yields a number that no longer certifies anything. Raw attribution tensors are not conserved; projecting them onto a conserved flow changes the object and that projection must be stated.
- **Strict positivity.** One zero-flow edge forces $f_P \le 0$, and Corollary 2 assumes $f > 0$ throughout. Such edges lie on no safe path and should be pruned first.
- **Enumeration is not fully algebraic.** Equation (14) gives the maximum-excess *value* per vertex pair, not the path realising it; recovering paths needs back-pointers, and characterising the family of all safe paths ending at a vertex needs Lemma 2(e), which has no matrix form here.
- **No asymptotic claim.** The tropical closure restates rather than accelerates: it is $O(nm)$ against the source paper's $O(mn)$ sweep. What the matrix form supplies is a single object, $\lambda$, from which verification, maximality, the all-pairs table, the gradient, the relaxation and §10 all follow without further argument.
- **Signed arithmetic.** $f_P$ is a difference, not a flow value, and is unbounded below; size integer widths against $\sum_i f_{\mathrm{out}}(u_i)$, not against $\max f$. In floating point the strict test $> 0$ is a boundary case, and a tie among minimal leaks in (9) must be broken explicitly for maximality to be well defined.
- **Version.** The results reformulated here are those of arXiv:2102.06480**v1**. The ESA 2022 successor and the RECOMB 2022 companion strengthen the enumeration bounds; neither affects Theorem 1.

---

## 12. Verification

`test_safeflow.py` — 16 tests, runnable as `python test_safeflow.py` or under `pytest`. Each is named for the result it checks.

| Test | Establishes |
|---|---|
| `proposition_1_internal_vertex_identity` | (4) against dense $T$ and $H$, on every window of the worked instance |
| `theorem_1_identity_and_nonnegativity` | $\lambda$ equals $(T^\top T - I)f$ formed densely; $\lambda \ge 0$; (5) matches Definition (3) |
| `theorem_1_matches_published_values` | all 15 chain excess flows of the source paper's worked instance |
| `proposition_2_lambda_is_the_sibling_sum` | (6) edge by edge, for both $\lambda$ and $\mu$ |
| `corollary_1_dual_form` | (7) against (5) |
| `theorem_1_survives_without_conservation_but_corollary_1_does_not` | Theorem 1 holds on a deliberately unconserved flow while Corollary 1 fails — and the cancellation subtlety noted in §4 |
| `corollary_2_subpath_closure` | monotonicity over every sub-window of the full chain |
| `corollary_3_incremental_updates` | the $-\lambda_e$ / $-\mu_e$ endpoint deltas |
| `proposition_3_maximality_against_brute_force` | the two $O(1)$ tests agree with explicit extension search over all safe paths |
| `the_three_chain_maximal_paths` | the maximal safe paths of the worked instance, with excess values |
| `e1_a_window_can_fail_locally_yet_extend_globally` | the enumeration defect of §5.1, on a concrete instance |
| `proposition_4_closure_matches_brute_force_and_the_naive_power` | (14) against exhaustive path enumeration **and** against the naive $(I\oplus L)^{\otimes(n-1)}$ product |
| `proposition_5_excess_never_exceeds_rollout` | $f_P \le R(P)$ over all paths of two graphs; equality on single edges |
| `proposition_6_leaks_are_complements_of_W_and_Delta` | (17); and that $W_e = 1/\delta_\to$, $\Delta_e = 1/\delta_\leftarrow$ exactly on a uniform graph |
| `proposition_7_uniform_collapse` | (18) exactly, and that only single edges are safe, for $d \in \{2,3,4\}$ |
| `proposition_7_boundary_out_degree_one_permits_long_paths` | tightness of the bound at $d = 1$ |

---

## 13. References

1. S. Khan and A. I. Tomescu. *Safety of Flow Decompositions in DAGs.* arXiv:2102.06480**v1**, Feb 2021. — Definition 1, Theorem 1 and Lemma 2 are this paper's; Propositions 1–7 and Corollaries 1–3 above are the present reformulation.
2. A. Vitvitskyi, J. G. M. Araújo, M. Lackenby, P. Veličković. *What Makes a Good Feedforward Computational Graph?* ICML 2025, PMLR 267; arXiv:2502.06751. — The walk and diffusion matrices, mixing time, minimax fidelity, and Propositions 4.2 and 5.1 cited in §10.
3. S. Khan and A. I. Tomescu. *Optimizing Safe Flow Decompositions in DAGs.* ESA 2022 / arXiv:2102.06480v2. — Optimal output-sensitive enumeration; supersedes v1's algorithmic bounds.
4. S. Khan, M. Kortelainen, M. Cáceres, L. Williams, A. I. Tomescu. *Safety and Completeness in Flow Decompositions for RNA Assembly.* RECOMB 2022 / arXiv:2201.10372.
5. S. Abnar and W. Zuidema. *Quantifying Attention Flow in Transformers.* ACL 2020. — Attention rollout, the sum-product reading of §8.
6. A. Mensch and M. Blondel. *Differentiable Dynamic Programming for Structured Prediction and Attention.* ICML 2018. — The softmin relaxation of §6.
