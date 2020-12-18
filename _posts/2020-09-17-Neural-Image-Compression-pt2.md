---
layout: post
title: "Neural Image Compression II"
date: 2020-11-12
categories: machine-learning, data-compression, variational-inference
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: Learned data compression may be cast in the language of variational inference. We'll study this analogy for lossy compression, which leads us to learnable compression schemes based on latent variable models, generalizing autoencoder-based methods. 
excerpt_separator: <!--more-->
---

This is the second post in an officially multi-part series about learnable data compression (you can find [Part I here]({% post_url /_posts/2020-09-12-Neural-Image-Compression %})) and investigates how data compression can be cast in the language of variational inference. For an overview of variational inference you can read [this previous post]({% post_url /_posts/2020-06-15-LVM-primer %}) - or perhaps you'd prefer to read an actual book. I recommend Blei. et. al. (2016) [[1]](#1) if you choose the latter route. <!--more-->

* Contents
{:toc}

# Variational Data Compression
There are multiple ways to cast data compression as a variational problem - that is, reformulating the problem as optimization of some variational bound through adjusting the parameters of some probability model. Here we will focus on the case of lossy compression. For observed data $$\*x$$ and additional posited latent variables $$\*z$$, recall that variational inference casts the problem of finding the true posterior $$p(\*z \vert \*x)$$ as optimization of the KL-divergence between the approximating posterior $q_{\phi}$ and true posterior, which corresponds to maximization of the negative free-energy/ELBO $$\mathscr{F}(\*x; \phi)$$ over $\phi$ (here we work in the amortized setting, where the distribution parameters of $q_{\phi}$ are input-dependent).

$$\begin{align*}
    \kl{q_{\phi}(\*z \vert \*x)}{p(\*z \vert \*x)} &= \log p(\*x) + \E{q_{\phi}}{\log q_{\phi}(\*z \vert \*x) - \log  p(\*x,\*z)} \\
    &= \log p(\*x) - \mathscr{F}(\*x; \phi) 
\end{align*}$$

## Rate-Distortion and the ELBO

Examining one possible form of the (negative) ELBO:

$$\begin{equation}
    -\mathscr{F}(x; \phi) = -\E{q_{\phi}(z \vert x)}{\log p(x \vert z)} + \kl{q_{\phi}(z \vert x)}{p(z)}
\end{equation}$$

The first term on the right, the negative log-likelihood of $x$ given samples $z \sim q_{\phi}$, can be interpreted as the per-sample distortion by rewriting the conditional model as:

$$\begin{equation}
    p(x \vert z) \propto \exp\left(- \lambda d\left(x,G_{\theta} \circ  \lfloor E_{\phi}(x) \rceil\right)\right).
\end{equation}$$

Selecting the conditional $p(x \vert z)$ to be a factorized Gaussian corresponds to the standard Euclidean distortion metric. However, note that not all distortion measures necessarily correspond to a normalized distribution (e.g. it's not clear (to me) how to formulate perceptual distortions in this manner). Alternatively, we can view this as the expected message length required to encode $x$, given a sample $z \sim q_{\lambda}$. This is applicable to the case for lossless reconstruction, but not lossy reconstruction, where the compressed latents $z$ are sent in lieu of the original data $x$.

Examining the second term, we note that the KL divergence decomposes as:

$$\begin{equation}
    \kl{q_{\phi}(z \vert x)}{p(z)} = -\E{q_{\phi}(z \vert x)}{\log p(z)} - \mathbb{H}(q_{\phi}).
\end{equation}$$

The first term, the cross-entropy between $q_{\lambda}$ and the prior $p(z)$ can be interpreted as the expected message length required to encode data drawn from $q_{\lambda}$ using an optimal code designed for $p(z)$. The entropy term $\mathbb{H}(q)$ is related to a compression scheme known as bits-back encoding that allows us to recover $\mathbb{H}(q)$ bits by an appropriate transmission protocol - see Townsend et. al. (2019) [[2]](#2) for a good overview. This procedure is only applicable to lossless encoding, so we can sweep this under the rug and neglect this term here. Note that for discrete latents the entropy is non-negative, and dropping $$\mathbb{H}(q_{\phi})$$ yields an upper bound on the negative ELBO. Alternatively, if the variational distribution has some fixed entropy, as can be achieved by fixing it's variance, then the rate-distortion loss is equal to the negative ELBO, up to a constant. This allows us to interpret the rate-distortion loss as:

$$\begin{equation}
    \label{eq:rd_primitive}
    \mathcal{L} = -\E{p(x)}{\E{q_{\phi}(z \vert x)}{\log p(x \vert z) + \log p(z)}}
\end{equation}$$

Where the first term has the interpretation of the distortion and the last has the interpretation as the rate. So minimization of the RD objective with a stochastic encoder (during training) is equivalent to maximization of the ELBO. This objective is optimized by finding an appropriate combination of the likelihood $p(x \vert z)$ and the variational posterior minimizing Eq. \ref{eq:rd_primitive}. For amortized inference under the VAE framework, this procedure involves learning the transforms from the original space to the latent space and vice-versa which parameterize the likelihood and the variational posterior.

## Discussion

Admittedly the analogy is not perfect and we've swept some stuff under the rug here - is using a stochastic encoder (during training) valid? Most latent variable models (LVMs) operate with a continuous latent space, not discrete - does the upper bound remain valid? Note that the stochasticity and continuous latents only apply during the LVM optimization process. At compression time we use a deterministic encoder to infer the latents, which are discretized before being passed to some entropy coder. Of course the question is why this works during optimization time. Ballè et. al. (2018) [[3]](#3) provide some empirical justification by arguing that certain types of additive noise can simulate quantization error and demonstrate that the learned continuous densities closely match the probability mass functions at integer locations, allowing the differential entropies to closely match the Shannon entropies of the latents.

To end, is the connection between variational inference and lossy compression just surface level? We can awkwardly massage the rate-distortion objective into the ELBO, but is there something more fundamental going on here? In the hypothetical next part of this series we'll try to answer this question in the affirmative by using the basics of information theory as our starting point.

<!-- ## 1. Bits-Back Coding

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

So we see that the expected message length under bits-back coding is equal to the free-energy $\mathscr{F}$ (or negative ELBO). So variational inference - maximizing the ELBO - can be interpreted as minimizing the expected message length using bits-back coding under the probabilistic models $$\{q_{\lambda}, p(\*z), p(\*x \vert \*z)\}$$. -->

<!-- ## 2. Entropy Model
The final probability model is a two-level hierarchial model where the latents are described by a Gaussian or logistic distribution with spatially adaptive parameters. The latents $$\*y$$ are assumed conditionally independent given the hyperlatents $$\*z$$. The latter is modelled using a non-parametric factorized density with the logits are predicted for each subpixel. This setup reflects a prior belief that the likelihood of neighbouring values in the latent obtain should be position-dependent - e.g. neighbouring values in the latent domain should have similar probability, while allowing a large degree of freedom for the hyperlatent values. -->

## References

<a id="1">[1]</a> 
David M. Blei, Alp Kucukelbir, Jon D. McAuliffe.,
"Variational Inference: A Review for Statisticians.",
arXiv:1601.00670 (2016).

<a id="2">[2]</a> 
James Townsend, Tom Bird, David Barber
"Practical Lossless Compression with Latent Variables using Bits Back Coding",
arXiv:1901.04866 (2019).

<a id="3">[3]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston.,
"Variational image compression with a scale hyperprior",
arXiv:1802.01436 (2018).

![grass](/assets/images/shell_cushion.jpg)
