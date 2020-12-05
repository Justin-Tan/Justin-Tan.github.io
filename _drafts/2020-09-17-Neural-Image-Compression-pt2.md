---
layout: post
title: "Neural Image Compression, Part II"
date: 2020-11-12
categories: machine-learning, data-compression, pytorch, variational-inference
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: Both lossy and lossless data compression may be cast in the language of variational inference. We'll study this analogy and show that it allows us to derive useful learnable compression schemes that can improve upon solely autoencoder-based methods. 
excerpt_separator: <!--more-->
---

This is the second post in an officially multi-part series about learnable data compression, and investigates how data compression can be cast in the language of variational inference. <!--more-->

* Contents
{:toc}

## 1. Bits-Back Coding

The above discussion can be framed in terms of probabilistic models of the observed $$\*x$$ and latent $$\*z$$. Suppose we have a sender who wishes to communicate observation $$\*x$$ to a receiver. Assume both have access to some parametric generative model $$q_{\lambda}(\*z \vert \*x)$$ and probabilistic models $$p(\*x \vert \*z)$$, $$p(\*z)$$. Then the bits-back coding procedure for the sender reads:

1. Sample latent $$\*z \sim q_{\lambda}(\*z \vert \*x)$$.
2. Encode/transmit the observed data via $$p(\*x \vert \*z)$$
3. Encode/transmit the sampled latent via $$p(\*z)$$.

The receiver does the inverse of steps 2 and 3 to recover $$(\*x,\*z)$$, requiring an expected message length of $$-\E{q_{\lambda}}{\log p(\*x \vert \*z) + \log p(\*z)}$$. The expectation of this term over the data distribution corresponds to the rate-distortion objective with a (possibly parametric) value of $\beta$, where $$p(\*x \vert \*z) \propto \exp\left(-d(\*x,\*x')\right)$$. In particular, if the model $$p(\*x \vert \*z)$$ is taken be a Gaussian likelihood, this corresponds to the mean square loss. The additional bits used by the sender are recovered by encoding the latent sample via the $$q_{\lambda}(\*z \vert \*x)$$. The expected code length of these extra bits is $$-\E{q_{\lambda}}{\log q_{\lambda}(\*z \vert \*x)}$$, and the receiver gets these 'bits back' through the common generative model $$q_{\lambda}(\*z \vert \*x)$$. This means that the expected transmitted message length will be:

$$\E{q_{\lambda}}{\log q_{\lambda}(\*z \vert \*x) - \log p(\*x,\*z)}$$

This looks pretty familiar! Time for a quick detour into variational inference - for observed data $$\*x$$ and additional posited latent variables $$\*z$$, recall that variational inference casts the problem of finding the true posterior $$p(\*z \vert \*x)$$ as optimization of the KL-divergence between the approximating posterior $q_{\lambda}$ and true posterior, which corresponds to maximization of the free-energy $$\mathscr{F}(\*x; \lambda)$$ over $\lambda$.

$$\begin{align}
    \kl{q_{\lambda}(\*z)}{p(\*z \vert \*x)} &= \log p(x) + \E{q_{\lambda}}{\log q_{\lambda}(\*z \vert \*x) - \log  p(\*x,\*z)} \\
    &= \log p(\*x) - \mathscr{F}(\*x; \lambda) 
\end{align}$$

So we see that the expected message length under bits-back coding is equal to the free-energy $\mathscr{F}$ (or negative ELBO). So variational inference - maximizing the ELBO - can be interpreted as minimizing the expected message length using bits-back coding under the probabilistic models $$\{q_{\lambda}, p(\*z), p(\*x \vert \*z)\}$$.

## 2. Entropy Model
The final probability model is a two-level hierarchial model where the latents are described by a Gaussian or logistic distribution with spatially adaptive parameters. The latents $$\*y$$ are assumed conditionally independent given the hyperlatents $$\*z$$. The latter is modelled using a non-parametric factorized density with the logits are predicted for each subpixel. This setup reflects a prior belief that the likelihood of neighbouring values in the latent obtain should be position-dependent - e.g. neighbouring values in the latent domain should have similar probability, while allowing a large degree of freedom for the hyperlatent values.

## References

<a id="1">[1]</a> 
Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur.,
"High-Fidelity Generative Image Compression.",
arXiv:2006.09965 (2020).

<a id="2">[2]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston.,
"Variational image compression with a scale hyperprior",
arXiv:1802.01436 (2018).

![grass](/assets/images/grass_again.jpg)
