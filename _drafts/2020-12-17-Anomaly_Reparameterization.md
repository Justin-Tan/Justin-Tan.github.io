---
layout: post
title: "Anomaly Detection and Invertible Reparameterizations"
date: 2020-12-17
categories: machine-learning, anomaly-detection
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: Density-based anomaly detection methods identify anomalies in the data as points with low probability density under the learned model. However, an invertible reparameterization of the data can arbitrarily change the density levels of any point - should anomalous points in one representation retain this status in any other representation?
excerpt_separator: <!--more-->
---

A recent [paper](https://openreview.net/pdf?id=6rTxPcEL7-) (Le Lan, Dinh, 2020) [[1]](#1) at the nicely themed ["I Can't Believe it's not Better" workshop](https://i-cant-believe-its-not-better.github.io/) at NeurIPS2020 asserts the following:

**Principle:** "_In an infinite data and capacity setting, the result of an anomaly detection method should be invariant to any continuous invertible reparametrization_."

In other words, the status of an "anomaly" in the observed data is a latent property associated with each datapoint, preserved across any choice of invertible reparameterization. We investigate arguments for and against this principle here.<!--more-->

* Contents
{:toc}

## Preliminaries
The task of anomaly detection is to partition the input space of the observed data $\mathcal{X}$ into the disjoint 'inlier' $$\mathcal{X}_{in}$$ and 'anomalous' $$\mathcal{A}$$ sets such that $$\mathcal{X} = \mathcal{X}_{in} \bigcup \mathcal{A}$$. Intuitively, we would expect anomalous points to belong to a subset $\mathcal{A} \subset \mathcal{X}$ with low probability mass. One natural way of defining this subset is using the true probability density $$p_X^{*}$$ over the observed data:

$$\begin{equation}
    \mathcal{A} = \{ x \in \mathcal{X} \; \vert \; p_X^{*}(x) \leq \lambda \}
\end{equation}$$

That is, anomalous points are points whose density falls below the threshold $\lambda$ chosen such that $\mathcal{A}$ covers a region of low probability mass. To practically identify points $x \in \mathcal{A}$, one may use generative models which approximate $$p_X^{*}$$ by some parametric model $p_X^{(\theta)}$. To avoid issues with model mis-specification, the paper works in the 'infinite data/capacity' setting, where the learnt density matches the ground truth exactly, $$p_X^{(\theta)} = p_X^{*}$$.

## Reparameterization

The main contention of the paper is that the status of anomaly-or-not is inherently attached to outcomes $\omega \in \Omega$ in some sample space, and that this status should remain invariant with respect to the choice of random variable $X: \omega \rightarrow \mathcal{X}$ mapping $\omega$ to the observed data. The authors argue that any invertible map $f: \mathcal{X} \rightarrow \mathcal{X}$ should not affect the anomaly-or-not status of $\omega$ - that is the observation $x = X(\omega)$ before reparameterization and the observation $f(x) = f \circ X(\omega)$ after reparameterization should be classified identically by an ideal anomaly detection algorithm.

This should be a contentious point, because density-based methods binds the definition of $\mathcal{A}$ to a particular choice of representation. From change of variables under the invertible map $f$ we have:

$$\begin{equation}
    p_{f(X)}(f(x)) = p_X(x) \cdot \left\vert \textsf{det}\left( J_f(x) \right) \right\vert^{-1}.
\end{equation}$$

Where $J_f$ denotes the Jacobian of the transformation. From this, it should be clear that density-based approaches for anomaly detection do not satisfy the above principle - one can subject $X$ to some sort of multivariate probability integral transform such that the $f(X)$ is uniformly distributed on the hypercube. Density-based anomaly detection in this parameterization can't draw any useful conclusions as $p_{f(X)}(f(x)) = 1 \, \forall \, x \in \mathcal{X}$.


## References

<a id="1">[1]</a> 
Le Lan, Charline and Dinh, Laurent.
"[Perfect density models cannot guarantee anomaly detection.](https://openreview.net/pdf?id=6rTxPcEL7-)",
I Can't Believe it's not Better Workshop, NeurIPS 2020.

<!-- ![grass](/assets/images/grass_again.jpg) -->
