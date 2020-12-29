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

In other words, the status of an "anomaly" in the observed data is a latent property associated with each outcome, preserved across any choice of invertible reparameterization. Whether or not you consider this principle realistic will depend on your definition of an anomaly.<!--more-->

* Contents
{:toc}

## Preliminaries
The task of anomaly detection is to partition the input space of the observed data $\mathcal{X}$ into the disjoint 'inlier' $$\mathcal{X}_{in}$$ and 'anomalous' $$\mathcal{A}$$ sets such that $$\mathcal{X} = \mathcal{X}_{in} \, \bigcup \, \mathcal{A}$$. Intuitively, we would expect anomalous points to belong to a subset $\mathcal{A} \subset \mathcal{X}$ with low probability mass. One natural way of defining this subset is using the 'true' probability density $$p_X^{*}$$ over the observed data:

$$\begin{equation}
    \label{eq:level_set}
    \mathcal{A} = \{ x \in \mathcal{X} \; \vert \; p_X^{*}(x) \leq \lambda \}
\end{equation}$$

That is, anomalous points are points whose density falls below the threshold $\lambda$ chosen such that $\mathcal{A}$ covers a region of low probability mass. To practically identify points $x \in \mathcal{A}$, one may use generative models which approximate $$p_X^{*}$$ by some parametric model $p_X^{(\theta)}$. To avoid issues with model mis-specification, the paper works in the 'infinite data/capacity' setting, where the learnt density matches the ground truth exactly, $$p_X^{(\theta)} = p_X^{*}$$.

## Reparameterization

The main contention of the paper is that the status of anomaly-or-not is inherently attached to outcomes $\omega \in \Omega$ in some sample space, and that this status should remain invariant with respect to the choice of random variable $X: \omega \rightarrow \mathcal{X}$ mapping $\omega$ to the observed data. The authors argue that any invertible map $f$ should not affect the anomaly-or-not status of $\omega$ - that is the observation $x = X(\omega)$ before reparameterization and the observation $f(x) = f \circ X(\omega)$ after reparameterization should be classified identically by an ideal anomaly detection algorithm.

Defining anomalies as points of low probability density fails to satisfy the invariance principle, as these methods bind the definition of $\mathcal{A}$ to a particular choice of representation. From change of variables under the invertible map $f$ we know:

$$\begin{equation}
    p_{f(X)}(f(x)) = p_X(x) \cdot \left\vert \textsf{det}\left( J_f(x) \right) \right\vert^{-1}.
\end{equation}$$

Where $J_f$ denotes the Jacobian of the transformation. Put another way, the density $\lambda$-level sets (Eq. $\ref{eq:level_set}$) are not necessarily preserved under an invertible transformation, and the anomaly/not-anomaly status becomes dependent on the parameterization. The figure below provides a slightly contrived example of how an invertible mapping changes the density associated with each point in the original representation.

<div class='figure'>
    <img src="/assets/plots/betas.png"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Points of low density in a given representation may not necessarily have low density in another representation. The two beta distributions to the left and right are related by the invertible map in the middle, given by $f(x) = F_{\beta}^{-1}\left(F_{\beta}(x; \; \alpha_1, \beta_1); \; \alpha_2, \beta_2\right)$, where $F_{\beta}$ is the CDF of the Beta distribution and $\alpha_i, \beta_i$ are their respective parameters.
    </div>
</div>

One can also subject $X$ to some sort of multivariate probability integral transform $$f: \mathcal{X} \rightarrow [0,1]^D$$ such that $f(X)$ is uniformly distributed on the hypercube. Density-based anomaly detection in this parameterization cannot draw any useful conclusions here as $p_{f(X)}(f(x)) = 1 \, \forall \, x \in \mathcal{X}$. The authors further proceed to show that, given knowledge of the true density $p_X^*$, it is possible (if impractical) to find a continuous mapping that assigns an arbitrary density to each datapoint $x \in \mathcal{X}$, which means the performance of any density-based anomaly detection method can be made arbitrarily bad, even in the infinite-capacity limit.

## Discussion

From this, it should be clear that approaches which identify anomalies via Eq. $\ref{eq:level_set}$, or related methods such as those based on weak typicality [[2]](#2), cannot satisfy the above invariance principle - it is possible to manipulate the information contained within the density in an adversarial manner such that the resulting anomaly detection method performs poorly.

However, we know that existing density-based anomaly detection methods are commonplace and relatively successful - so there exist a large range of representations which are both easily learnable and separate inliers from outliers well through their learned densities. The natural question is which choice of random variable $X$ leads to the best anomaly detection performance? This cuts to the heart of how to define an anomaly. It seems natural to define anomalies using the original input representation. That is, using Eq. $\ref{eq:level_set}$, where $$p_X^*$$ is the true data-generating distribution of the input. From this perspective, in the infinite data/capacity limit, estimating $$p_X^*$$ would be guaranteed to succeed at detecting every anomaly.

However, in the real, finite-capacity world, there is no guarantee that this would be ideal. Much of the success of deep learning is predicated on representation learning; downstream tasks typically operate on a highly processed function of the original representation. The same may hold for anomaly detection - it is likely there exists an alternative representation where anomalies are easier to separate from non-anomalies. Furthermore, according to the manifold hypothesis, the effective dimensionality of the data may be much smaller than the ambient dimensionality, and estimating the density in the original given representation is likely to be inefficient.

Alternatively, if we consider anomaly status to be a latent variable attached to each outcome, as the paper suggests, then the problem becomes analogous to binary classification. Using this definition of anomaly, the invariance principle becomes more reasonable, but still a very strong assumption. In this case, using the original input data representation is not guaranteed to be ideal, even in the infinite-capacity setting. This is because it is not possible to determine whether a point is drawn from one of two distributions with overlapping support, and there may exist an alternate parameterization in which the anomaly and non-anomaly distributions are better separated than the original input representation.

### Another Anomalous Set Definition

A more general but non-constructive way of defining the set of anomalies $\mathcal{A}$ which subsumes Eq. $(\ref{eq:level_set})$ is as a set of outcomes $\omega$ of low probability mass. That is, $$\mathbb{P}\left(X(\omega) \in \mathcal{A}\right) \leq \alpha$$ for some small $\alpha$. We know any invertible transformation $f$ should be volume-preserving:

$$\begin{align*}
\int_{x \in \mathcal{A}} dx \, p_X(x) &= \int_{y \in f(\mathcal{A})} dy \, p_{f(X)}(y) \\
\implies \mathbb{P}\left(X(\omega) \in \mathcal{A}\right) &= \mathbb{P}\left(f \circ X(\omega) \in f\left(\mathcal{A}\right)\right).
\end{align*}$$

So this provides a reparameterization-invariant definition of the set of anomalies, but gives no mechanism to construct $\mathcal{A}$.

## Summary

The paper illustrates nicely why density-based methods for anomaly detection are not guaranteed to accurately identify anomalies via an interesting impossibility result, which holds even if we are working in the infinite-capacity limit. My main takeaways are:

* Density based methods for anomaly detection are inherently coupled to a particular choice of representation.
* The original input representation is not necessarily the ideal space to measure likelihoods in.
* Fully agnostic anomaly detection methods can be misled by reparameterization and a useful definition of anomalies requires the injection of problem-dependent prior knowledge.
* This prior knowledge should be used to inform the design/application of AD methods to identify useful representations, whether learned or provided - preferably one in which irrelevant information is abstracted away.

I believe that the invariance principle put forward by the authors may be too strong a condition for useful anomaly detection methods. Forcing your algorithm to output predictions invariant to arbitrary invertible reparameterization seems too restrictive, and it's not clear what benefit you could expect from this over a method that works well in certain representations but is totally confused in others.

 Instead, a problem-dependent relaxation of the principle may offer a useful starting point. In situations where we have knowledge of the symmetries the problem should satisfy, it would likely be helpful to directly inject this domain knowledge into our algorithm, such that the result of the AD method is invariant to a problem-dependent subclass of invertible reparameterizations. I'm not sure if this will outperform just throwing the deepest generative model you have at the problem, but this may help bias our methods toward learning useful representations so that we may successfully apply Eq. $(\ref{eq:level_set})$ for anomaly detection.

## References

<a id="1">[1]</a> 
Le Lan, Charline and Dinh, Laurent. "[Perfect density models cannot guarantee anomaly detection.](https://openreview.net/pdf?id=6rTxPcEL7-)",
I Can't Believe it's not Better Workshop, NeurIPS 2020.

<a id="2">[2]</a> 
Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Balaji Lakshminarayanan. "[Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality.](https://arxiv.org/abs/1906.02994)", arXiv:1906.02994 (2019).

![grass](/assets/images/shell_zzZZz.jpg)
