---
layout: post
title: "Latent Variable Models in Jax"
date: 2020-06-15
categories: machine-learning, latent-variables, jax
---

## Latent Variable Models for Density Estimation in Jax
In this post we'll recap the basics of latent variable models for density estimation. Two approaches will be highlighted - the importance weighted autoencoder and the recently proposed SUMO (Stochastically Unbiased Marginalization Objective), an unbiased estimator of the log-marginal likelihood. We'll implement both these models in Jax, and see how Jax differs from standard Autodiff frameworks. Finally, we end with the obligatory illustrative toy problem. There's a nice picture of my neighbour's cat at the end for some extra motivation.

- Latent Variable Models
- Importance Weighting: Motivation
- SUMO
- Implementation in `Jax`
- Illustrative Toy Problem - "Neal's Funnel"
- Conclusion

## Latent Variable Models

Latent variable models allow us to augment observed data $x \in \mathcal{X}$ with unobserved, or latent variables $z \in \mathcal{Z}$. This is a modelling choice, in the hopes that we can better describe the observed data in terms of unobserved factors. We do this by postulating a conditional model $p(x \vert z)$ that describes the dependence of observed data on latent variables, as well as a prior density $p(z)$ over the latent space. Typically defining this conditional model requires some degree of cleverness, or we can do the vogue thing and just use a big neural network.

Latent variables are useful because they may reveal something useful about the data. For instance, we may postulate that the high-dimensionality of the observed data may be 'artificial', in the sense that the observed data can be explained by more elementary, unobserved variables. Informally, we may hope that these latent factors capture most of the relevant information about the high-dimensional data in a succinct low-dimensional representation. To this end, we may make the modelling assumption that $$\text{dim}\left(\mathcal{Z}\right) \ll \text{dim}\left(\mathcal{X}\right)$$.

This is a test $\int_x p(y \vert x) p(x) $

Also a test \\[\int \pi /6 \\]

