---
layout: post
title: "Latent Variable Models in Jax"
date: 2020-06-15
---

## Hi
In this post we'll recap the basics of latent variable models for density estimation. In particular we'll look at two approaches - the importance weighted autoencoder and the recently proposed SUMO (Stochastically Unbiased Marginalization Objective), an unbiased estimator of the log-marginal likelihood. An implementation of each in Jax will be presented, together with the obligatory illustrative toy problem. There's a nice picture of my neighbour's cat at the end if you need some extra motivation.

* Latent Variable Models
* Importance Weighting: Motivation
* SUMO
* Implementation in Jax
* Illustrative Toy Problem - "Neal's Funnel"
* Conclusion
