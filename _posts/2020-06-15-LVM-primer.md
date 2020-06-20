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

## **Black-Box Variational Inference**
Even if we are able to efficiently compute or approximate well the \elbo, we wish to maximize this estimate with respect to the variational parameters $\lambda$. So during this optimization procedure, we wish to compute the gradient of an expectation under some distribution with respect to the same parameters of said distribution:
 
$$ \begin{equation}
     \nabla_{\lambda}  \E{q_{\lambda}(z)}{\log p(x,z) - \log q_{\lambda}(z)}
 \end{equation}$$
 
As the distribution depends on the parameters $\lambda$, we cannot simply interchange the gradient and integral. We wish to phrase the integrand in a way that allows us to do this. This problem is more widespread than variational inference, and appears in reinforcement learning during computation of the policy gradient and sensitivity analysis in financial markets as well. This gradient estimation problem carries with it all the associated challenges of evaluating the expectation, plus now we have to compute derivatives. Black-box variational inference (BBVI) sidesteps evaluation of intractable expectations by expressing the gradient of the ELBO with respect to the variational parameters $\lambda$ as an expectation with respect to some density that may be efficiently sampled from. The gradient may then be approximated by Monte Carlo. We detail three of the main approaches to this problem below for optimization of the expectation of some objective function $f(z; \lambda)$.
 
### Score Function Estimator
The name of this technique arises from the following identity regarding the score function, the derivative of the log of a distribution with respect to the distribution parameters:

$$\begin{equation}
    \nabla_{\lambda} q_{\lambda}(z) = q_{\lambda}(z) \nabla_{\lambda} \log q_{\lambda}(z)
\end{equation}$$
 
 Assume uniform convergence and continuity of $f$ to interchange the order of the limiting process and integration - this is usually taken for granted, as most objective functions are differentiable almost everywhere and not too weird.
 
$$\begin{align}
     \nabla_{\lambda} \E{q_{\lambda}(z)}{f(z; \lambda)} &= \nabla_{\lambda} \int dz\;  q_{\lambda}(z) f(z; \lambda) \\
     &= \int dz \; \nabla_{\lambda} q_{\lambda}(z) f(z; \lambda) + q_{\lambda}(z)\nabla_{\lambda}f(z; \lambda) \\
     &= \int dz \;  q_{\lambda}(z) \left(\nabla_{\lambda} \log q_{\lambda}(z) f(z;\lambda) + \nabla_{\lambda} f(z;\lambda)\right)\\
     &= \E{q_{\lambda}(z)}{\nabla_{\lambda}\log q_{\lambda}(z) f(z;\lambda) + \nabla_{\lambda}f(z;\lambda)} \\
     &\approx \frac{1}{K} \sum_{k=1}^K \nabla_{\lambda} \log q_{\lambda}(z) f(z_k; \lambda) + \nabla_{\lambda} f(z_k; \lambda), \quad z_k \sim q_{\lambda}
 \end{align}$$
 
 However, this estimator is not used in practice because it has a large variance, leading to slow convergence during optimization. To see this, note that the gradient of the score function has zero mean:
 
 $$\begin{align}
     \int dz \; q_{\lambda}(z) \nabla_{\lambda} \log q_{\lambda}(z) &= \int dz \; \nabla_{\lambda} q_{\lambda}(z) \\
     &= \nabla_{\lambda} \int dz \; q_{\lambda}(z) \\
     &= \nabla_{\lambda} (1) = 0.
 \end{align}$$
 
 So if $f(z; \lambda)$ can take on large values - which is possible during the early stages of training if $f$ is something like a log-likelihood, then the Monte Carlo estimate of the gradient will oscillate wildly around zero, making convergence difficult. There exist numerous variance-reduction technqiues to make the score function estimator usable. 
 
### Reparameterization
 Sometimes also known as the pathwise estimator. The core idea is that, for certain classes of distributions, samples from $q_{\lambda}(z)$ may be expressed as $\lambda$-dependent deterministic transformations from some base distribution which is independent of $\lambda$:
 
 $$\begin{equation}
     \hat{z} \sim q_{\lambda}(z) \quad \equiv \quad \hat{\epsilon} \sim p(\epsilon), \; \hat{z} = g(\hat{\epsilon}; \lambda)  
 \end{equation}$$
 
 This method is less general than the score function estimator, as it requires the transformation/path $g$ to exist, but enjoys lower variance of the gradient estimator in general. One famous example of this is expressing samples from a general multivariate Gaussian $q_{\lambda}(z) = \mathcal{N}\left(z \vert \mu, \Sigma\right)$ as a transformation of samples from the standard Gaussian $p(\epsilon) = \mathcal{N}(z \vert \mathbf{0}, \mathbf{1})$. where the Cholesky decomposition of the covariance matrix $\Sigma = L L^*$:
 
 $$\begin{equation}
     \hat{z} \sim q_{\lambda}(z) \quad \equiv \quad \hat{\epsilon} \sim p(\epsilon), \; \hat{z} = \mu + L\hat{\epsilon} 
 \end{equation}.$$
 
Reparameterization relies on the ability to express the expectation over some distribution $q_{\lambda}(z)$ parameterized by $\lambda$ by replacing $z \leftarrow g(\epsilon; \lambda)$ and converting to an expectation over the base distribution $\epsilon$, provided the transformation $g$ is differentiable and invertible:

$$\begin{equation}
\E{q_{\lambda}(z)}{f(z;\lambda)} = \E{p(\epsilon)}{f\left(g(\epsilon;\lambda); \lambda\right)}
\end{equation}$$

The justification for this is sometimes called the Law of the Unconscious Statistician (LOTUS) - 'Unconscious' is apt, as we already used this previously when assuming the expected value $\E{p(x)}{f(x)} = \int dx \; p(x) f(x)$. A short proof sketch is given in the Appendix.
 
The LOTUS allows us to pull the gradient operator into the integral after the change of variables, allowing us to obtain an estimator of the gradient of $f(z; \lambda)$ immediately,

$$\begin{align}
     \nabla_{\lambda} \E{q_{\lambda}(z)}{f(z; \lambda)} &= \nabla_{\lambda} \E{p(\epsilon)}{f(g(\epsilon, \lambda); \lambda)} \\ 
     &= \E{p(\epsilon)}{\nabla_{\lambda} f(g(\epsilon, \lambda); \lambda)} \\
     &\approx \frac{1}{K}\sum_{k=1}^K \nabla_{\lambda} f(g(\hat{\epsilon}_k, \lambda); \lambda), \quad \hat{\epsilon}_k \sim p(\epsilon)
\end{align}$$

Reparameterization rocketed to prominence owing to its use in variational autoencoders and normalizing flows, where the base distribution $p(\epsilon)$ is typically taken to be the standard Gaussian, and the transformation $g$ is represented by some sort of neural network architecture. Here the variational parameters $\lambda$ are not learnt per-datapoint but taken to be the parameters of $g$.

## **What can I do with $\log p(x)$?**
Assume that our primary objective is distribution modelling - we are not interested in the latent variables for the time being, treating them as some sort of mathematical sleight of hand to efficiently evaluate the objective $\log p(x)$. Estimating this objective has two main applications. Below let the true data distribution be $p^*(x)$ and make explicit the model parameters by writing the estimate as $p_{\theta}(x)$.

### Density Estimation
Here we would like to find a reliable estimate of the log-probability of a given observation $x$. This can be achieved by minimizing the 'forward' KL divergence between the true distribution and the model distribution:

$$\begin{align}
    \mathcal{L}(\theta) = \kl{p^*(x)}{p_{\theta}(x)} &= \E{p^*(x)}{\log \frac{p^*(x)}{p_{\theta}(x)}} \\
    &= -\E{p^*(x)}{\log p_{\theta}(x)} - \mathbb{H}(p^*) 
\end{align}$$

The expectation is taken over the target distribution, so the forward-KL is useful where the target density cannot be explicitly evaluated, but samples from the target distribution are easily accessible/generated. This is the case in most popular applications, e.g. image processing, language modelling. Note that minimization of the forward-KL with respect to $\theta$ reduces to minimization of the cross-entropy between the target and model distribution, as the entropy of $p^\*(x)$ is a constant term, so maximizing the ELBO is equivalent to minimizing the forward KL. If $p_{\theta}(x)$ is close to zero in a region where $p^\*(x)$ is nonzero, this will incur a large penalty under the forward KL divergence - thus models trained using the forward KL will be fairly liberal with density assignment in $x$-space in the sense that they will overestimate the support of $p^\*(x)$.

It is also possible to minimize the 'reverse' KL divergence, which interchanges the arguments of the forward KL divergence: 

$$\begin{align}
    \mathcal{L}(\theta) = \kl{p_{\theta}(x)}{p^*(x)} &= \E{p_{\theta}(x)}{\log \frac{p_{\theta}(x)}{p^*(x)}} \\
    &= \E{p_{\theta}(x)}{\log p_{\theta}(x) - \log p^*(x)} \\
    &= -\mathbb{H}(p_{\theta}) - \E{p_{\theta}(x)}{\log p^*(x)}.
\end{align}$$

Unlike the forward-KL, reverse-KL requires explicit evaluation of the (possibly unnormalized) target density $p^*(x)$ (the normalization constant enters as a constant term in the objective function with respect to $\theta$), so this is suitable if we have access to the true target density but no efficient sampling algorithm. We also need to be able to efficiently sample from the model to generate the Monte Carlo estimates used during optimization. 

Note that the reverse-KL objective can be minimized by maximization of the entropy $\mathbb{H}(p_{\theta})$, which corresponds to minimization of $\log p_{\theta}(x)$. This is in the 'morally wrong direction' to how we are optimizing the \textsc{elbo} - we are attempting to minimize a lower bound. This is problematic as minimization of $\log p_{\theta}(x)$ can be achieved through artificial deterioration of the bound, optimizing $\theta$ to maximize the gap $\kl{q_{\lambda}(z)}{p(z\vert x)}$ between the \textsc{elbo} and $\log p_{\theta}(x)$ instead of minimizing the true objective. In contrast to the forward KL, the reverse KL incurs a large penalty if $p_{\theta}(x)$ is nonzero wherever $p^\*(x)$ is close to zero, so models trained under the reverse KL will be more conservative with probability mass and underestimate the support of $p^\*(x)$.

### Approximate Sampling 
Another application is to generate samples from the target distribution $p^*(x)$ using the model $p_{\theta}$ as a surrogate. This is achieved by sampling from the approximate posterior in latent space and subsequently sampling from the conditional model:

$$\begin{equation}
    x \sim p_{\theta}(x) \equiv z \sim q_{\lambda}(z), \quad x \sim p_{\theta}(x \vert z)
\end{equation}$$

## Appendix: LOTUS Proof

Here we make explicit the random variable each distribution is defined over, defining $q_{\lambda} = p_Z$ and dropping the $\lambda$-dependence for clarity:

- Let the cumulative distribution function (CDF) of $z$ be $F_Z(z) = \mathbb{P}(Z \leq z)$, similarly for $\epsilon$: $F_{\varepsilon} = \mathbb{P}(\varepsilon \leq \epsilon)$, then note:

$$\begin{align}
    F_Z(z) &= \mathbb{P}(Z \leq z) \\
    &= \mathbb{P}(g^{-1}(Z) \leq g^{-1}(z)) \\
    &= \mathbb{P}(\epsilon \leq g^{-1}(z)) \\
    &= F_{\varepsilon}(g^{-1}(z))
\end{align}$$

- Then using the definition of the CDF and change of variables under the transformation $t = g(t')$:

$$\begin{align}
    \int_{-\infty}^z dt \; p_Z(t) &= \int_{-\infty}^{g^{-1}(z)} dt' \; p_{\varepsilon}(t') \\
    &= \int_{-\infty}^z dt \; p_{\varepsilon}(g^{-1}(t)) \vert \det \nabla_{t'} g(t') \vert^{-1} \\
    &= \int_{-\infty}^z dt \; p_{\varepsilon}(g^{-1}(t)) \vert \det \nabla_{g^{-1}(t)} g(g^{-1}(t)) \vert^{-1}
\end{align}$$

- Differentiating w.r.t. to $z$, noting that $\epsilon = g^{-1}(z)$:

$$\begin{equation}
    p_Z(z) = p_{\varepsilon}(g^{-1}(z)) \vert \det \nabla_{\epsilon} g(\epsilon) \vert^{-1} 
\end{equation}$$

- Now note that, by change of variables under the transformation $\epsilon = g^{-1}(z)$, see e.g. Baby Rudin, 10.9, and substitution of our earlier expression for $p_Z(z)$, we finish the proof off.

$$\begin{align}
    \E{p_{\varepsilon}}{f(g(\epsilon))} &= 
    \int d\epsilon \; p_{\varepsilon}(\epsilon) f\left(g(\epsilon)\right) \\
    &= \int dz \; p_{\varepsilon}\left(g^{-1}(z)\right) f(z) \vert \det \nabla_{\epsilon} g(\epsilon)\vert^{-1} \\
    &= \int dz \; p_Z(z) f(z)\\
    &= \E{p_Z(z)}{f(z)}
\end{align}$$
