---
layout: post
title: "Concentration Inequalities I: Tensorization Identities"
type: note
date: 2020-12-04
categories: machine-learning, information-theory
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: Tensorization allows us to break apart complicated joint distributions of random variables to a sum of simpler terms related to their marginals. Here we cover a range of useful tensorization identities which may be used to derive interesting concentration inequalities. 
excerpt_separator: <!--more-->
---

The following is a collection of useful results which I'm writing down because I can't remember them. This will lay the foundation for future parts, where we'll see how this can be used to derive interesting concentration inequalities. The content here loosely follows Boucheron et. al. (2012) [[1]](#1) and Duchi (2019) [[2]](#2).  <!--more-->

* Contents
{:toc}

## 1. Preliminaries - Chain Rules

Tensorization identities are based on chain rules for measures of information and divergences. The most basic chain rule is the following decomposition of the joint entropy which can be seen using log-laws and the chain rule for probability (using $X_{<n}$ to denote the sequence of r.vs from $X_1$ through $X_{n-1}$):

$$\begin{align}
    H(X_1, \ldots, X_N) &= \sum_{x_n \in \mathcal{X}} p(x_1, \ldots, x_N) \log \prod_{n} p(x_n \vert x_{<n}) \nonumber \\
    &= \sum_n H\left(X_n \vert X_{<n}\right) \leq \sum_n H(X_n).
\end{align} $$

Where the last 'sub-additivity' inequality follows as conditioning reduces entropy, on average. The chain rule for mutual information is a direct consequence of this:

$$\begin{align}
    I(X; Y_1, \ldots, Y_N) &= H(Y_1, \ldots, Y_N) - H(Y_1, \ldots, Y_N \vert X) \nonumber \\
    &= \sum_n \left[ H(Y_n \vert Y_{<n}) - H(Y_n \vert X, Y_{<n})\right] = \sum_n I(X; Y_n \vert Y_{<n}).
\end{align}$$

Assuming the $Y_n$s are conditionally independent given $X$ (e.g. commonly assumed for the latent variables in a VAE, or any situation sampling from a distribution parameterized by $X$), leads to the following inequality:

$$\begin{align}
    I(X; Y_1, \ldots, Y_N) &= \sum_n \left[ H(Y_n \vert Y_{<n}) - H(Y_n \vert X, Y_{<n})\right] \nonumber \\
    &=  \sum_n \left[ H(Y_n \vert Y_{<n}) - H(Y_n \vert X) \right] \leq \sum_n \left[ H(Y_n)- H(Y_n \vert X) \right] = \sum_n I(X; Y_n).
\end{align}$$

### KLD chain rules
The chain rule for the KL divergence between the product distributions of independent r.v.s $P = \prod_i P_i$ and $Q = \prod_i Q_i$ is immediate from log-laws:

$$\begin{equation}
    \kl{P}{Q} = \sum_i \kl{P_i}{Q_i}
\end{equation}$$

For the more general case for arbitrary joint distributions $P$ and $Q$ over the r.v.s $$\{X_i\}_{i=1}^N$$, we first define the conditional KL divergence between conditional distributions:

$$\begin{equation*}
    \kl{P_{Y \vert X}}{Q_{Y \vert X}} = \sum_{x \in \mathcal{X}} p(x) \sum_{y \in \mathcal{Y}} p(y \vert x) \log \frac{p(y \vert x)}{q(y \vert x)}
\end{equation*}$$

Then using the chain rule of probability we may derive a general chain rule for the relative entropy between $p$ and $q$:

$$\begin{align}
    \kl{P}{Q} &= \sum_{x \in \mathcal{X}} \left[\prod_{m=1}^N p(x_m \vert x_{<m})\right] \log \prod_{n=1}^N \frac{ p(x_n \vert x_{<n})}{q(x_n \vert x_{<n})} \nonumber \\
    &= \sum_{n=1}^N \sum_{x \in \mathcal{X}} \left[\prod_m p(x_m \vert x_{<m})\right] \log \frac{ p(x_n \vert x_{<n})}{q(x_n \vert x_{<n})} \nonumber \\
    &= \sum_{n=1}^N \sum_{x_{<n} \in \mathcal{X}} p(x_{<n}) \sum_{x_n \in \mathcal{X}} p(x_n \vert x_{<n}) \log \frac{ p(x_n \vert x_{<n})}{q(x_n \vert x_{<n})} \nonumber \\
    &= \sum_{n=1}^N \kl{P_{X_n \vert X_{<n}}}{Q_{X_n \vert X_{<n}}}
\end{align}$$

Where in the third line we note marginalizing out variables with index $>n$ doesn't affect the terms in the sum with index $\leq n$. It's probably easier to prove this for the two-variable case $\kl{P_{XY}}{Q_{XY}}$ and then induct.

## 2. Han's Inequalit(ies)

### Han's Inequality

Another direct consequence of the chain rule for the joint entropy is Han's Inequality, useful for deriving concentration inequalities. Let $$X_{\setminus n} = \{X_i\}_{i=1}^N \setminus \{X_n\}$$ be the vector of r.v.s obtained by leaving out $X_n$ from the total collection $(X_1, \ldots, X_N)$, then, using the definition of conditional entropy: $H(X_i \vert X_j) = H(X_i, X_j) - H(X_j)$:


$$\begin{align*}
    H(X_1, \ldots, X_N) &= H(X_n \vert X_{\setminus n}) + H(X_{\setminus n}) \\
    &\leq H(X_n \vert X_{<n}) + H(X_{\setminus n}) && \textsf{(Conditioning reduces entropy)}
\end{align*}$$

There are $N$ such inequalities for each $n = 1,\ldots, N$, taking their sum yields:

$$\begin{align}
    N H(X_1, \ldots, X_N) &\leq \sum_n \left(H(X_n \vert X_{<n}) + H(X_{\setminus n})\right) \nonumber \\
    &\leq H(X_1, \ldots, X_N) + \sum_n H(X_{\setminus n}) && \textsf{(Chain rule for joint entropy)} \nonumber \\
    \implies H(X_1, \ldots, X_N) &\leq \frac{1}{N - 1} \sum_n H(X_{\setminus n})
\end{align}$$

### Han's Inequality for Relative Entropies
Let $\mathcal{X}$ denote some finite alphabet to which each of the r.vs $(X_1, \ldots, X_N)$ belong. Let $P$ and $Q$ be distributions over $\mathcal{X}^N$. $Q$ may be arbitrary, but $P = \prod_i P_i$ is a product distribution over the space. This may seem restrictive, but $P$ will usually be taken to be a distribution over i.i.d. r.v.s. Define $P^{(\setminus n)}$ and $Q^{(\setminus n)}$ to be the corresponding distributions obtained by marginalization over $X_n$, with corresponding PMFs:

$$\begin{align*}
    q^{(\setminus n)}(x_{\setminus n}) &= \sum_{x \in \mathcal{X}} q(x_1, \ldots, x_{n-1}, x, x_{n+1}, \ldots, x_N) \\ 
    p^{(\setminus n)}(x_{\setminus n}) &= \prod_{i \neq n} p(x_i)
\end{align*}$$

Then Han's Inequality for relative entropies says:

$$\begin{equation}
    \kl{Q}{P} \geq \frac{1}{N-1} \sum_{n=1}^N \kl{Q^{(\setminus n)}}{P^{(\setminus n)}}
\end{equation}$$

There are a lot of strange indices in the proof, but it follows relatively clearly from the first Han's Inequality we proved above and the definition of the KL divergence. See Chapter 4 of Boucheron et. al. [[1]](#1) if you're interested.

## 3. Tensorization of the 'Entropy'

To begin, for a convex function $\phi: \mathbb{R} \rightarrow \mathbb{R}$ we define the $\phi$-entropy.

$$\begin{equation*}
    \mathbb{H}_{\phi}(X) \triangleq \E{}{\phi(X)} - \phi\left(\E{}{X}\right)
\end{equation*}$$

As a special case, we can choose $\phi(t) = t \log t$, to obtain what we will simply call the 'Entropy'. To distinguish this from the Shannon entropy, we capitalize the 'E' and denote the symbol by a blackboard-H:

$$\begin{equation*}
    \mathbb{H}(Z) \triangleq \E{}{Z \log Z} - \E{}{Z}\log \E{}{Z}
\end{equation*}$$

To connect this with concentration inequalities, the intuition is if $Z$ is concentrated around it's mean, $Z \approx \E{}{Z}$ and the 'Entropy' should be small.

The final section of this post is concerned with finding a tensorization identity for the 'Entropy' of $Z = f(X_1, \ldots, X_N)$ to control $\mathbb{H}(Z)$. Here $$\{X_i\}_{i=1}^N $$ are i.i.d. r.v.s drawn from the product distribution $P = \prod_i P_i$ as previously and $f: \mathcal{X}^N \rightarrow \mathbb{R}^+$ is a non-negative function of the $X_n$s.

Define $\mathbb{H}\left(Z \vert X_{\setminus n}\right) = \mathbb{E}^{(n)}\left[Z \log Z\right] - \mathbb{E}^{(n)}[Z] \log \mathbb{E}^{(n)}[Z]$ to be the 'Entropy' with the expectation taken over $X_n$ only (here $\mathbb{E}^{(n)}$ denotes expectation with respect to $X_n$ only - i.e. a conditional expectation conditioned on $X_{\setminus n}$). The following theorem says we can control the 'Entropy' of $Z$ by controlling the individual conditional 'Entropies':

### Sub-additivity of the 'Entropy'

$$\begin{equation}
    \mathbb{H}[Z] \leq \mathbb{E}\left[\sum_{n=1}^N \mathbb{H}\left(Z \vert X_{\setminus n}\right) \right].
\end{equation}$$

*Proof:*
The above inequality holds if and only if it holds for a positive rescaling $cZ, c>0$ due to the linearity of the expectation, so assume without loss of generality that $\E{}{Z}=1$, killing off the second term in the 'Entropy' - now $\mathbb{H}(Z) = \E{}{Z \log Z}$.

Introduce the tilted distribution $Q$ on $\mathcal{X}^N$ with PMF $$q(\*x) = f(\*x)p(\*x), \; \*x \in \mathcal{X}^N$$, then the 'Entropy' can be written as the relative entropy between $Q$ and $P$.

$$ \kl{Q}{P} = \E{\*x \sim Q}{\log \frac{q(\*x)}{p(\*x)}} = \E{\*x \sim P}{f(\*x) \log f(\*x)} = \E{\*x \sim P}{Z \log Z}.$$

The same holds for the conditional 'Entropy'. Denote the conditional expectation of $Z$ w.r.t. only $X_n$ as:

$$\begin{equation*}
\E{x \sim P_n}{Z \vert x_{\setminus n}} = \sum_{x \in \mathcal{X}} p_n(x) f\left(x_1, \ldots, x_{n-1}, x, x_{n+1}, \ldots x_N\right)
\end{equation*}$$

The relative entropy between the distributions $$Q^{(\setminus n)}$$ and $$P^{(\setminus n)}$$ with $$X_n$$ marginalized out becomes:

$$\begin{align*}
\kl{Q^{(\setminus n)}}{P^{(\setminus n)}} &= \sum_{x_{\setminus n} \in \mathcal{X}^{n-1}} p^{(\setminus n)}(x_{\setminus n}) \, \E{x \sim P_n}{Z \vert x_{\setminus n}} \log \frac{p^{(\setminus n)}(x_{\setminus n}) \left[ \sum_{x \in \mathcal{X}} p_n(x) \, \E{x \sim P_n}{Z \vert x_{\setminus n}} \right]}{p^{(\setminus n)}(x_{\setminus n})} \\
&= \E{x_{\setminus n} \sim \prod_{i \neq n} P_i}{\phi\left( \E{x \sim P_n}{Z \vert x_{\setminus n}} \right)}.
\end{align*}$$

Where we let $\phi(t) = t \log t$. The law of iterated expectation allows us to relate the two KL terms:

$$\begin{align*}
    \kl{Q}{P} - \kl{Q^{(\setminus n)}}{P^{(\setminus n)}} &= \E{\*x \sim P}{\phi(Z)} - \E{x_{\setminus n} \sim \prod_{i \neq n} P_i}{\phi\left( \E{x \sim P_n}{Z \vert x_{\setminus n}} \right)} \\
    &= \E{x_{\setminus n} \sim \prod_{i \neq n} P_i}{ \E{x \sim P_n}{\phi(Z) \vert X_{\setminus n}} - \phi\left( \E{x \sim P_n}{Z \vert x_{\setminus n}} \right)} \\
    &= \E{x_{\setminus n} \sim \prod_{i \neq n} P_i}{\mathbb{H}\left(Z \vert X_{\setminus n}\right)}
\end{align*}$$


This allows us to apply Han's Inequality for relative entropies to yield the result:

$$\begin{align*}
\mathbb{H}[Z] = \kl{Q}{P} &\leq \sum_{n=1}^N \left( \kl{Q}{P} - \kl{Q^{(\setminus n)}}{P^{(\setminus n)}} \right) \\
&= \sum_{n=1}^N \E{x_{\setminus n} \sim \prod_{i \neq n} P_i}{\mathbb{H}\left(Z \vert X_{\setminus n}\right)} \\
&= \E{}{\sum_{n=1}^N \mathbb{H}\left(Z \vert X_{\setminus n}\right)}
\end{align*}$$

The proof is simpler than it looks, I chose to make the distribution each expectation is taken with respect to clear, which cluttered things a bit, Chapter 5 of Duchi (2019) [[2]](#2) provides a not-too-much-longer more general proof for continuous $X_n$.

With the dry stuff out of the way we can move on to practical applications in deriving concentration inequalities in later posts.

## References

<a id="1">[1]</a> 
Boucheron, S., Lugosi, G., Massart, P.,
"Concentration Inequalities - A nonasymptotic theory of independence.",
Clarendon Press (2012).

<a id="2">[2]</a> 
Duchi, J.
"Lecture Notes for Statistics 311/Electrical Engineering 377",
Stanford (2019).

<!-- ![grass](/assets/images/grass_again.jpg) -->
