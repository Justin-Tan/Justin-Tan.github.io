---
layout: post
title:  "Latent Variable Model Primer"
date:   2020-06-15 16:01:44 +1000
categories: machine-learning, latent-variables
usemathjax: true
readtime: True
image: /assets/images/shell_web1.png
excerpt_separator: <!--more-->
---

This post is meant to serve as a quick primer on latent variable models. We'll walk through a straightforward derivation of the evidence lower bound and then examine how this can be used for density estimation / sampling.<!--more--> If you're already familiar with this material, only the last section may be of interest to you. 
* Contents
{:toc}

## **1. Latent Variable Models**

Latent variable models allow us to augment observed data $x \in \mathcal{X}$ with unobserved, or latent variables $z \in \mathcal{Z}$. This is a modelling choice, in the hopes that we can better describe the observed data in terms of unobserved factors. We do this by defining a conditional model $p(x \vert z)$ that describes the dependence of observed data on latent variables, as well as a prior density $p(z)$ over the latent space. Typically defining this conditional model requires some degree of cleverness, or we can do the vogue thing and just use a big neural network.

Latent variables are useful because they may reveal something useful about the data. For instance, we may postulate that the high-dimensionality of the observed data may be 'artificial', in the sense that the observed data can be explained by more elementary, unobserved variables. Informally, we may hope that these latent factors capture most of the relevant information about the high-dimensional data in a succinct low-dimensional representation. To this end, we may make the modelling assumption that $$\text{dim}\left(\mathcal{Z}\right) \ll \text{dim}\left(\mathcal{X}\right)$$.

A natural question is to invert the conditional model - given the observed $x$, what can we learn about unobserved $z$? - to this end, we are interested in finding the posterior $p(z \vert x)$:

$$\begin{equation}
    p(z \vert x) = \frac{p(x \vert z)p(z)}{p(x)}.
\end{equation}$$

For instance, we may want to sample from $p(z \vert x)$ or compute expectations of some function $f$ under the posterior $\E{p(z \vert x)}{f(z)}$.  This usually requires us to compute the marginal $p(x)$ by integrating out the unobserved variables:

$$\begin{equation}
    \label{eq:marginal_px}
    p(x) = \int_{\mathcal{Z}} dz \; p(x \vert z) p(z).
\end{equation}$$

But this is difficult to do! The more complex your conditional $p(x \vert z)$ is, the worse of a time you'll have trying to evaluate this. This integral has a time complexity exponential in $\text{dim}(\mathcal{Z})$ even for a simple conditional like a mixture of Gaussians, and analytic solutions and tractability go out the window the second you introduce neural networks into the mix.

## **2. Approximating $p(z \vert x)$**
To circumvent the direct computation of $p(x)$, it is common to approximate the model posterior by a simpler distribution - one that allows efficient evaluation and sampling. Denote this approximating distribution with parameters $\lambda$ as $q_{\lambda}(z)$. Then we would like to find the distribution $q_{\lambda}(z)$ in some restricted family of 'nice' distributions $\mathcal{Q}$ that is as close as possible to the true posterior in terms of some sort of discrepancy function between distributions. The canonical choice is the 'reverse' KL divergence $\kl{q_{\lambda}(z)}{p(z \vert x)}$:

\begin{equation}
\label{eq:vi_obj}
\lambda^* = \argmin_{\lambda} \E{q_{\lambda}(z)}{\log \frac{q_{\lambda}(z)}{p(z \vert x)}}
\end{equation}

The final optimized variational distribution $q_{\lambda^*}(z)$ may then be used in lieu of the true posterior. 

## **2.1. Optimizing for $q_{\lambda}$**
Unfortunately, we do not have access to the true posterior $p(z \vert x)$, so cannot directly optimize $\kl{q_{\lambda}(z)}{p(z \vert x)}$. Nevertheless, applying Bayes' Theorem to the objective in Equation \ref{eq:vi_obj} allows us to define an objective in terms of quantities we can evaluate:

$$\begin{align}
    \kl{q_{\lambda}(z)}{p(z \vert x)} &=  \E{q_{\lambda}(z)}{\log q_{\lambda}(z) -  \log  p(z \vert x)} \\
    &= \E{q_{\lambda}(z)}{\log q_{\lambda}(z) - \left(\log  p(x,z) - \log p(x) \right)} \label{eq:form2_step}\\
    &= \log p(x) + \E{q_{\lambda}(z)}{\log q_{\lambda}(z) - \log  p(x,z)} \\
    &= \log p(x) - \mathscr{F}(x; \lambda) 
\end{align}$$

Where the variational free energy $\mathscr{F}$ is defined as:

\begin{equation}
    \label{eq:elbo_def}
    \mathscr{F}(x; \lambda) = \int dz \; q_{\lambda}(z) \left(\log p(x,z) - \log q_{\lambda}(z)\right).
\end{equation}

This is a lower bound on $\log p(x)$ due to the non-negativity of the KL-divergence, hence it is also common to refer to $\mathscr{F}(x; \lambda)$ as the Evidence Lower BOund, or ELBO (which is a more useful mnemonic). We'll use these terms interchangeably throughout our discussion (it's awkward to write $\mathscr{F}$ in text mode and awkward to write ELBO in math mode).

\begin{equation}
    \log p(x) = \kl{q_{\lambda}(z)}{p(z \vert x)} + \mathscr{F}(x; \lambda)
\end{equation}

As $\log p(x)$ is a constant with respect to the variational parameters $\lambda$, minimizing $\kl{q_{\lambda}(z)}{p(z \vert x)}$ is equivalent to maximization of $\mathscr{F}(x; \lambda)$. This leads us to two ways of thinking about optimization of $\mathscr{F}$:

- Optimization of a lower bound on $\log p(x)$. This is useful for modelling the distribution of the observed data. In this case the latent variables $z$ are a sort of mathematical sleight of hand for estimation of the density $p(x)$. 
- Inference of $p(z \vert x)$ - we specify some augmented model $p(x,z)$ with hidden variables $z$ and are interested in approximating the posterior $p(z \vert x)$. In this case we are interested in what the latent variables can tell us about the structure of our observed data. 

Note we write $\mathscr{F}$ out as an integral explicitly to generate some apprehension - if we wish to optimize the lower bound analytically, this limits the modelling distributions to simple distributions where it is possible to evaluate the integral in closed form. Alternatively, evaluating this with numerical quadrature generally scales exponentially in the number of latent variables and may be unrealistically computationally expensive. 