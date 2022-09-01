---
layout: post
title: "Monte Carlo Methods and Normalizing Flows"
date: 2020-08-04
type: note
categories: machine-learning, normalizing-flows, monte-carlo, jax
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
subtitle: Importance sampling can be used to reduce the variance of Monte Carlo integration, including computing expectations. We investigate the main idea and then examine an interesting application of normalizing flow models to this area.
---

Here we study an interesting application of normalizing flows to Monte Carlo integration.<!--more--> The first two sections deal with basics about Monte Carlo and importance sampling, and the third section explores how flow models can be used to define a low-variance importance sampling density. 

<!-- The last section looks at how some difficult integrals can be approximated in Jax. -->

* Contents
{:toc}

## 1. Monte Carlo

Monte Carlo (MC) integration aims to approximate an expectation by the mean of a randomly drawn sample of random variables. Let $$\*x \in \mathcal{X} \subseteq \mathbb{R}^D$$ be some random variable with distribution $$p(\*x)$$. Assuming we are working in the continuous domain, the expectation of some function, $f: \mathcal{X} \rightarrow \mathbb{R}$ is:

$$\begin{equation}
\E{p}{f(\*x)} = \int_{\mathcal{X}} d\*x \, p(\*x) \, f(\*x).
\end{equation}$$

If we take an i.i.d. sample $$(\*x_1, \ldots, \*x_N)$$ drawn from $$p(\*x)$$, then we define the _Monte Carlo estimate_ of the expectation as the sample mean (note this is an unbiased estimator):

$$\begin{equation}
\bar{f}_N(\*x) = \frac{1}{N}\sum_{n=1}^N f(\*x_n).
\end{equation}$$

The weak law of large numbers says that the MC estimate becomes arbitrarily close to the true value, within some $\epsilon > 0$, as we increase the number of samples $N$:

$$\begin{equation}
\lim_{N \rightarrow \infty} \mathbb{P}\left( \left\vert \; \bar{f}_N(\*x) - \E{p}{f(\*x)} \right\vert  \geq \epsilon\right) = 0.
\end{equation}$$

The variance of the MC estimate scales as $1/N$, irrespective of the number of dimensions, making this an attractive estimator for high-dimensional integrals:

$$\begin{equation}
\mathbb{V}_p\left[\bar{f}(\*x)\right] = \frac{1}{N}\mathbb{V}_p\left[ f(\*x) \right] = \frac{1}{N} \int_{\mathcal{X}} d\*x \, p(\*x) \left( f(\*x) - \E{p}{f(\*x)}\right)^2
\end{equation}$$

Integration problems can be naively converted to Monte Carlo problems by specifying an integration domain $\Omega$ and writing the integral as an expectation under the uniform distribution over $\Omega$:


$$ \begin{equation}
\int_{\Omega} d\*x \; f(\*x) = \vert \Omega \vert \E{\mathcal{U}(\Omega)}{f(\*x)} \approx \frac{\vert \Omega \vert}{N} \sum_{n} f(\*x_n), \quad \*x_n \sim \mathcal{U}(\Omega).
\end{equation}$$


This is inefficient for high dimensional problems as the integrand $$f(\*x)$$ may only assume non-negligible values in special regions of $\Omega$.

## 2. Importance Sampling
Importance sampling is one way of reducing the variance of the MC estimate $\bar{f}$. The idea is to use a distribution other than $$p(\*x)$$ from which to generate the points $$(\*x_1, \ldots, \*x_N)$$ from that focuses on 'important' regions where the integrand assumes non-negligible values. In fact, the integral we are interested in, $$\int_{\mathcal{X}} d\*x \, f(\*x) \, p(\*x)$$ can be represented in an infinite number of ways by the triplet $(f,p,\mathcal{X})$, and we can switch between triples through change of variables. One sensible way to proceed is to focus on those choices of triples that enable a low-variance MC estimate $\bar{f}_N$.

### 2.1. Example
Take the following example, from Robert and Casella [[1]](#1). Here we are interested in finding the exceedance $\geq 2$ of the Cauchy distribution $\mathcal{C}(0,1)$.

$$ p = \int_2^{\infty} \frac{dx}{\pi (1+x^2)}$$

Naively, we can sample $x_n$ i.i.d. from $\mathcal{C}(0,1)$ and use the MC estimator:

$$ \hat{p}_1 = \frac{1}{N} \sum_n \mathbb{I}\left[x_n \geq 2\right]; \quad x_n \sim \mathcal{C}(0,1).$$

The summand is a Bernoulli r.v. so the variance of this estimator is $\mathbb{V} = p(1-p)/N \approx 0.127/N$. Another representation of $p$ is obtained through change of variables $x \rightarrow y^{-1}$.

$$ p = \int_0^{1/2} dy \, \frac{y^{-2}}{\pi (1 + y^{-2})} = \frac{1}{2} \E{\mathcal{U}(0,\frac{1}{2})}{\frac{y^{-2}}{\pi (1 + y^{-2})}}.$$

In this form, $p$ is the expectation over the uniform distribution over $(0,1/2)$ and the MC estimator becomes:

$$ \hat{p}_2 = \frac{1}{N} \sum_n \frac{y_n^{-2}}{\pi(1+y_n^{-2})}; \quad y_n \sim \mathcal{U}(0,1/2).$$

Integration by parts shows that the variance $\mathbb{V} = \frac{1}{N} 9.5 \times 10^{-5}$. So evaluating expectations based off simulated r.vs from the original distribution may be inefficient - particularly for heavy-tailed distributions, due to simulated values being generated outside the relevant domain - $(2,\infty)$, in this case.

### 2.2. Estimator
Importance sampling decouples the expectation from the distribution $p$, introducing a _proposal distribution_ $q$, that is easy to sample and evaluate, and satisfies $\textsf{supp}(p) \subset \textsf{supp}(q)$:

$$ \E{p}{f(\*x)} = \int_{\mathcal{X}} dx \, f(\*x) \frac{p(\*x)}{q(\*x)} q(\*x) = \E{q}{\frac{f(\*x)p(\*x)}{q(\*x)}}. $$ 

The core idea is to choose the proposal $q$ to avoid drawing samples that lie outside the region where the integrand is non-negligible. The importance sampling estimator converges almost surely to the expectation of interest provided the condition on their respective supports is satisfied. However the variance will depend strongly on the choice of proposal:

$$ \begin{align}
\mathbb{V}_q\left[\frac{f(\*x)p(\*x)}{q(\*x)}\right] &= \E{q}{\left(\frac{f(\*x)p(\*x)}{q(\*x)}\right)^2} - \left( \E{q}{\frac{f(\*x)p(\*x)}{q(\*x)}} \right)^2 \\
&= \E{q}{\left(\frac{f(\*x)p(\*x)}{q(\*x)}\right)^2} - \left(\E{p}{f(\*x)}\right)^2.
\end{align}$$

Immediately we see that the ratio $p/q$ should not be too large to avoid the estimator from receiving large shocks, e.g. from sampled points where $$q(\*x) \ll p(\*x)$$. A formal result for the minimum-variance proposal is as follows:

$$\begin{equation}
    \argmin_q \mathbb{V}_q\left[\frac{f(\*x)p(\*x)}{q(\*x)}\right] = \argmin_q \E{q}{\left(\frac{f(\*x)p(\*x)}{q(\*x)}\right)^2}
\end{equation}$$

Minimizing this functional via the variational principle (see the Appendix) w.r.t $q$ demonstrates that the choice with lowest variance is:

$$ q^*(\*x) = \frac{\vert f(\*x) \vert p(\*x)}{\int_{\mathcal{X}} d\*y \, \vert f(\*y) \vert p(\*y)} $$

This makes intuitive sense - we want to sample from 'important' regions where $$p$$ or $$ \vert f \vert $$ are large and make a significant contribution to the integral. So $$q$$ should also be correspondingly large at these points. If $$f(\*x) \geq 0$$, then the variance of the importance sampling estimator $$\mathbb{V}_{q^*}$$ goes to zero, so we get the correct answer for any number of terms, $N$. This isn't practically useful because $$\int_{\mathcal{X}} d\*x \, f(\*x) p(\*x)$$ is the integral we're interested in - if we could evaluate the normalization factor in $q^*$ we can solve our original problem. However it does suggest that we should search for proposal distributions $q$ which are proportional, or 'have the same shape' as $\vert f p \vert $, i.e. $q \approx \frac{1}{Z} \vert f p \vert$, as distributions are unique up to normalization. 

So some informal desiderata for the proposal $$q(\*x)$$ are that:

- $q$ should have larger support than $fp$, that is $$q(\*x) > 0 \; \forall \; \{\*x \; \vert \; f(\*x)p(\*x) \neq 0\}$$. To avoid issues in tail regions, $q$ should be heavy-tailed.
- $q$ should be approximately proportional to the absolute value of the integrand $$\vert p(\*x)f(\*x) \vert$$.
- $q$ supports efficient sampling and evaluation.

The previous discussion extends to the case of estimating integrals - instead of sampling from the uniform distribution over $\Omega$, now the evaluation points are sampled from the proposal $q$:

$$\begin{align}
\int_{\Omega} d\*x \, f(\*x) &= \int d\*x \, \frac{f(\*x)}{q(\*x)} q(\*x) \\
&= \E{q}{\frac{f(\*x)}{q(\*x)}} \\
&\approx \frac{1}{N} \sum_n \frac{f(\*x_n)}{q(\*x_n)}, \quad \*x_n \sim q(\*x)
\end{align}$$

This estimator has zero variance if $$q(\*x) = \frac{f(\*x)}{\int_{\Omega} d\*x \; f(\*x)}$$, as can be verified by substitution into the above MC estimator.

## 3. Minimizing Variance using Flow Models
There is a long history of constructing proposals chosen from some parametric family $$\mathcal{Q} = \{ q_{\lambda} \vert \lambda \in \Lambda\}$$ minimizing some divergence measure between $q_{\lambda} \in \mathcal{Q}$ and the minimum variance estimator $q^*$. The standard approach is to minimize the KL divergence with respect to $\lambda$:

$$\argmin_{\lambda} \kl{q^*}{q_{\lambda}} = \argmax_{\lambda} \E{q^*}{\log q_{\lambda}(\*x)}$$

If $\mathcal{Q}$ is not sufficiently simple then this program may not be convex and we rely on gradient-based optimization:

$$ \begin{align}
\nabla_{\lambda} \kl{q^*}{q_{\lambda}} &= -\nabla_{\lambda} \int_{\mathcal{X}} d\*x \, q^*(\*x) \log q_{\lambda}(\*x) \\
 &= -\int_{\mathcal{X}} d\*x \, q_{\lambda}(\*x) \frac{q^*(\*x)}{q_{\lambda}(\*x)} \nabla_{\lambda} \log q_{\lambda}(\*x) \\
 &= -\E{q_{\lambda}}{\frac{q^*(\*x)}{q_{\lambda}(\*x)} \nabla_{\lambda} \log q_{\lambda}(\*x)}
\end{align}$$

Note that $q^*$ is only known up to the constant factor $$F = \int_{\mathcal{X}} d\*x \; \vert f(\*x) \vert p(\*x)$$, which can be absorbed into the (adaptive) learning rate of your favorite gradient-based optimization scheme.

If the variance is directly minimized, the gradient of the variance assumes a very similar form, except the weights of the score function are squared, and up to the constant factor $F^2$.

$$ \begin{align}
\mathbb{V}_{q_{\lambda}}\left[\frac{f(\*x)p(\*x)}{q_{\lambda}(\*x)}\right] = \E{q_{\lambda}}{\left(\frac{f(\*x)p(\*x)}{q_{\lambda}(\*x)}\right)^2} - \left(\E{p}{f(\*x)}\right)^2.
\end{align}$$

$$ \begin{align}
\nabla_{\lambda} \mathbb{V}_{q_{\lambda}}\left[\frac{f(\*x)p(\*x)}{q_{\lambda}(\*x)}\right] &= \int_{\mathcal{X}} d\*x \; \nabla_{\lambda} \frac{\left(f(\*x)p(\*x)\right)^2}{q_{\lambda}(\*x)} \\
&= -\int_{\mathcal{X}} d\*x \; \nabla_{\lambda}q_{\lambda}(\*x) \left(\frac{f(\*x)p(\*x)}{q_{\lambda}(\*x)}\right)^2 \\
&= -\int_{\mathcal{X}} d\*x \; q_{\lambda}(\*x) \nabla_{\lambda}\log q_{\lambda}(\*x) \left(\frac{f(\*x)p(\*x)}{q_{\lambda}(\*x)}\right)^2 \\
&= -\E{q_{\lambda}}{\left(\frac{f(\*x)p(\*x)}{q_{\lambda}(\*x)}\right)^2 \nabla_{\lambda}\log q_{\lambda}(\*x)} \\
&= -F^2\E{q_{\lambda}}{\left(\frac{q^*(\*x)}{q_{\lambda}(\*x)}\right)^2 \nabla_{\lambda}\log q_{\lambda}(\*x)}
\end{align}$$

One popular choice of family $\mathcal{Q}$ is a heavy-tailed Student's-$t$ mixture model $$\sum_k \pi_k t_{\nu_k}\left(\*x \vert \mathbf{\mu}_k, \mathbf{\Sigma}_k)\right)$$, where the learnable parameters are $$\left(\{\pi_k, \nu_k, \mathbf{\mu}_k, \mathbf{\Sigma}_k\}_{k=1}^K\right)$$. 

There are more expressive methods at hand; the generative modelling literature gives us multiple ways to construct complicated distributions as a parametric transformation of a simpler base distribution. The approach taken by flow-based models is to seek an invertible and differentiable parametric function $f_{\lambda}$ which maps from a random variable $$\*z$$ with a simple base density $$q_Z(\*z)$$ to the observed data $$\*x$$ (being explicit about random variables). The inverse image of this map provides a density model $$q_X(\*x)$$ for the original space through change of variables:

$$\begin{align}
q_X(\*x) = q_Z\left((f_{\lambda}^{-1}(\*x)\right) \; \left\vert \det \frac{\partial f_{\lambda}^{-1}(\*x)}{\partial \*x} \right\vert = q_Z(\*z) \; \left\vert \det \frac{\partial f_{\lambda}(\*z)}{\partial \*z} \right\vert^{-1}
\end{align}$$

As differentiability and invertibility are closed under composition, one can stack a sequence of transformations $$f_{\lambda} = f_{\lambda}^{(1)} \circ \ldots \circ f_{\lambda}^{(K)}$$ to produce a composite flow model that allows more expressive transformations to be learnt. Provided we can evaluate $$q_Z$$ and compute the determinant of the Jacobian of the transformation efficiently, we can take $q_{\lambda} = q_X$ to be the resulting density model. While the base density $q_Z$ is usually taken to be the standard normal, for importance-sampling based applications, it may be helpful to choose $q_Z$ to be heavy-tailed, e.g. using the Cauchy distribution or standard Student's-$t$ distribution with $\nu > 1$.

In summary, flow-based models offer an attractive way to approximate the minimum-variance distribution with $q_{\lambda}$, by optimizing the parameters of the transformation $$f_{\lambda}$$ to minimize $$ \kl{q^*}{q_{\lambda}}$$. This method appears to have been first proposed by Müller et. al., 2018 [[2]](#2), who apply this to calculating path integrals in light transport.

## References

<a id="1">[1]</a> 
Casella, George and Robert, Christian.
Monte Carlo Statistical Methods.
Springer (2004).

<a id="2">[2]</a> 
Müller, Thomas et. al.
Neural Importance Sampling.
arXiv 1808.03856 (2018).
