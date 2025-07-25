# Optimized Conformal Prediction

This project implements optimized conformal prediction methods for classification tasks. The repository is structured to facilitate experimentation, analysis, and deployment of conformal prediction techniques.

## Problem Background
Data $(x, y) \in \mathcal{X} \times \mathcal{Y}$, where $\mathcal{Y} = [K]$:

- Randomized prediction set $\Gamma$, such that
  $$
  \forall x \in \mathcal{X}, \quad \Gamma(x) \in [0,1]^{K}
  $$
- $[\Gamma(x)]_y = \mathbb{P}(y \in C(x))$ according to the randomness in $C$
- Can recover deterministic $C$: $\Gamma(x) \in \{0,1\}^{K}$

Coverage:
$$
\mathbb{P}(Y \in C(X)) = \mathbb{E}_{X} \left[ \langle \Gamma(X), p(\cdot|X) \rangle \right],
$$

### M-criterion

Consider
$$
\begin{align*}
&\min_{\Gamma} 
\mathbb{P}_{X,C} \left[ |C(X)| > 1 \right]
\\
&\mathrm{s.t. }~ \mathbb{E}_{C} \left[ \langle \Gamma(X), p(\cdot|X) \rangle\mid X=x \right] \ge 1-\alpha.
\end{align*}
$$

Now $|C(x)| \overset{d}{=} \sum_y \text{Ber}(\Gamma(x)_y)$, where the summands are independent. So,
$$
\left\{
\begin{aligned}
&P(|C(x)| = 0)  = \prod_y (1 - \Gamma(x)_y)\\
&P(|C(x)| = 1) = \sum_y  \Gamma(x)_y \prod_{y' \ne y} (1-\Gamma(x)_{y'})
\end{aligned}
\right.
$$

For fixed $x$, the objective becomes:
[$g$ is prob of 0,1]
$$
\operatorname{maximize}_{\gamma_i} g(\gamma)=\prod_{i=1}^K (1 - \gamma_i)+\sum_{i=1}^K \gamma_i \prod_{j \ne i} (1-\gamma_j)
$$
subject to
$$
\sum_{i=1}^Kp_i\gamma_i \geq 1-\alpha, 0\leq \gamma_i\leq 1,\ i=1,...,K
$$


## Conformalization
Once we have an estimated $\hat{\Gamma}$ based on an estimated $\hat{p}$, we need a way to conformalize it to achieve finite sample $1-\alpha$ coverage.

Here are two approaches:

- **Conformity Score with $\hat{\Gamma}$**:
  Use $\hat{\Gamma}(y|x)$ as the conformity score:
  $$
  C_\tau(x) := \{y \in \mathcal{Y} : \hat{\Gamma}(y|x) \geq \tau\},
  $$
  where $\tau$ is chosen to ensure the desired coverage. This corresponds to standard split conformal prediction.

- **Cumulative Sum Conformity Score**:
  Define the cumulative sum conformity score:
  $$
  s(x, i_{(j)}) = \sum_{r=1}^j \hat{\Gamma}_{(r)}(x), \quad j = 1, \ldots, K,
  $$
  where $i_{(j)}$ represents the indices sorted by $\hat{\Gamma}(x)$ in descending order. The prediction set is then given by:
  $$
  C_{\tau}(x) = \{i_{(j)} : s(x, i_{(j)}) \geq \tau\} = \{i_{(1)}, \ldots, i_{(m)}\},
  $$
  where $t$ is a threshold ensuring the desired coverage.