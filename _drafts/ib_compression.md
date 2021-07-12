---
layout: post
title: "IB compression"
date: 2021-01-25
type: note
categories: machine-learning, generalization
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: On compression.
excerpt_separator: <!--more-->
---

Consider information bottleneck (IB) framework with different priors for the latent space representation. Decompose the mutual information terms in the IB objective and estimate using variational approximations. <!--more-->

* Contents
{:toc}

## Preliminaries

* Joint generative distribution $$p_{\theta}(\*x, \*z) = p_{\theta}(\*x) p_{\theta}(\*x \vert \*z)$$. 
* Model data marginal $$p_{\theta}(\*x) = \a{p_{\theta}(\*x \vert \*z)}{p_{\theta}(\*z)}$$.
  * True empirical data distribution is $$p_D(\*x)$$.
* Inference model $$q_{\phi}(\*z \vert \*x)$$ with aggregated latent space marginal distribution $$q_{\phi}(\*z) = \a{q_{\phi}(\*z \vert \*x)}{p_D(\*x)}$$.



## References


<!-- ![grass](/assets/images/shell_cushion.jpg) -->
