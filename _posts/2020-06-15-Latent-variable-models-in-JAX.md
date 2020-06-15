---
layout: post
title: "Latent Variable Models in Jax"
subtitle: For density estimation/sampling
date: 2020-06-15
categories: machine-learning, latent-variables, jax
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

In this post we'll examine applications of LVMs to density estimation/fast sampling. We'll implement two illustrative approaches in Jax, highlighting how Jax differs from standard Autodiff frameworks in some major aspects.<!--more--> In particular, we'll look at the importance weighted autoencoder and the recently proposed SUMO (Stochastically Unbiased Marginalization Objective). Finally, we end with the obligatory illustrative toy problem. There's a nice picture of my neighbour's cat at the end for some extra motivation. If you'd rather read code, check out the [associated repository on Github](https://github.com/justin-tan/density_estimation_jax). 

- Latent Variable Models
- The Importance Weighted ELBO
- SUMO
- Implementation in `Jax`
- Application to efficient sampling from "Neal's Funnel"
- Conclusion

* Contents
{:toc}


## **1. Importance Weighted Autoencoders**
### **1.1 Heuristic Motivation**
Let $R$ be a random variable such that $\mathbb{E}\left[R\right] = p(x)$. Then we have:

\begin{equation}
    \log p(x) = \mathbb{E}\left[\log R\right] + \mathbb{E}\left[\log p(x) - \log R\right]
\end{equation}

Using Jensen's Inequality, we can interpret the first term on the right as a loewr bound on $\log p(x)$, and the second term as the bias. If we assume that $R$ stays relatively close to $p(x)$, then we can expand $\log R$ in a Taylor Series around $p(x)$:

$$\begin{align}
    \E{}{\log R} &= \E{}{\log\left(p(x) + R - p(x)\right)} \\
    &\approx \E{}{\log p(x) + \frac{1}{p(x)}\left(R - p(x)\right) - \frac{1}{2p(x)^2} \left(R - p(x)\right)^2} \\
    &= \log p(x) - \frac{1}{2p(x)^2} \mathbb{V}\left[R\right]
\end{align}$$

So the 'slackness' in the bound (approximately) scales with the variance of $R$, $\mathbb{V}\left[R\right]$:

\begin{align}
    \log p(x) - \mathbb{E}\left[\log R\right] \approx \frac{1}{2p(x)^2}\mathbb{V}\left[R\right] \geq 0
\end{align}

Note to make use of this expansion, we should have that $R$ is concentrated near $p(x)$ in the sense that the Taylor series for $\log\left(1 + \frac{R-p(x)}{p(x)}\right)$ converges, as:

$$ \log(\mu + R - \mu) = \log \mu + \log\left(1 + \frac{R-\mu}{\mu}\right) $$

This suggests we should look for a function of $R$ with the same mean but lower variance, one easy possibility is just the sample mean $R_K = \frac{1}{K}\sum_{k=1}^K R_k$. Then, as each $R_k$ has identical mean $p(x)$, we still have $\log p(x) \geq \E{}{\log R_K}$, but now this gives a tighter bound by a factor of $1/K$, with the caveat that $\vert \frac{R-p(x)}{p(x)} \vert < 1$. 

### **1.2. The Importance-Weighted ELBO**
We can connect this back to variational inference by letting $R$ have the following form:

\begin{equation}
    R = \frac{p(x,z)}{q(z)}, \quad z \sim q
\end{equation}

This satisfies the condition $\E{q}{R} = p(x)$. By Jensen's Inequality, we have the following lower bound on the log marginal likelihood:

\begin{equation}
    \E{q}{\log R} \leq \log p(x)
\end{equation}

From our heuristic argument above, we can tighten this bound by sampling multiple $z_k \sim q$ and using the sample mean over $R$ to estimate $p(x)$, arriving at the importance-weighted ELBO, or the IW-ELBO \citep{burda:iwae}. Being a tighter variational bound, this is a useful but biased replacement for $\log p(x)$ for density estimation problems. 

$$\begin{align}
    % R_K &= \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q \\
    \textrm{IW-ELBO}_K(x) &= \log \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q
\end{align}$$


## 2. Density Estimation
$$\begin{multline}
\begin{aligned}
G_t&=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 R_{t+4}+...\\
&=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4})+...\\
&=R_{t+1}+\gamma G_{t+1}\\
\end{aligned}
\end{multline}$$

$$\begin{align}
I(x;z) &= H(z) - H(z \vert x) \\
&= \pi/6
\end{align}$$

{% highlight python %}
def diag_gaussian_entropy(mean, logvar):
    # Entropy of diagonal Gaussian
    D = mean.shape[0]
    diff_entropy = 0.5 * (D * jnp.log(2 * jnp.pi) + jnp.sum(logvar) + D)
    return diff_entropy
{% endhighlight %}

### 1.2 Sampling


## 2. Importance-Weighted Autoencoders

### 2.1 Motivation

## 3. SUMO

### 3.1 Reducing Variance

