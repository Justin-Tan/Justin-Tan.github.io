---
layout: post
title: "Latent Variable Models in Jax"
date: 2020-06-15
---

## Hi
In this post we'll recap the basics of latent variable models for density estimation. Two approaches will be highlighted - the importance weighted autoencoder and the recently proposed SUMO (Stochastically Unbiased Marginalization Objective), an unbiased estimator of the log-marginal likelihood. We'll implement both these models in Jax, and see how Jax differs from standard Autodiff frameworks. Finally, we end with the obligatory illustrative toy problem. There's a nice picture of my neighbour's cat at the end for some extra motivation.

* Latent Variable Models
* Importance Weighting: Motivation
* SUMO
* Implementation in Jax
* Illustrative Toy Problem - "Neal's Funnel"
* Conclusion
