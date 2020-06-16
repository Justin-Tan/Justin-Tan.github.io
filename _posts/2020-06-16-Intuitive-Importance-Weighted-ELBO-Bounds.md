---
layout: post
title: "Intuitive IWAE Bounds"
subtitle: "Pumping those rookie numbers up."
date: 2020-06-16
categories: machine-learning, latent-variables
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

Here we'll give a quick and dirty sketch about why importance-weighted autoencoders provide a lower bound on the log-marginal likelihood $\log p(x)$ in a way that's slightly different from the original paper. <!--more-->


## **1. Importance Weighted Autoencoders**
Recall that the setup is that we are attempting to model the log-marginal likelihood $\log p(x)$ over some given data $x$. The evidence lower bound (ELBO) lower bounds (surprise!) $\log p(x)$ with gap given by the KL divergence between the learnt approximation $q_{\lambda}(z)$ and the true posterior $p(z \vert x)$. We want to maximize the lower bound with respect to the variational parameters $\lambda$ of $q_{\lambda}$. For more information check out this [earlier primer post]({% post_url 2020-06-15-LVM-primer %}).

$$\begin{align}
\kl{q_{\lambda}(z)}{p(z \vert x)} &=  \E{q_{\lambda}(z)}{\log q_{\lambda}(z) -  \log  p(z \vert x)} \\
    &= \log p(x) + \underbrace{\E{q_{\lambda}(z)}{\log q_{\lambda}(z) - \log  p(x,z)}}_{\text{ELBO}(x)}
\end{align}$$

Of course the natural question is if we can reduce this gap, and the answer happens to be yes, without too much effort.

### **1.1 Heuristic Motivation**
Let $R$ be a random variable such that $\mathbb{E}\left[R\right] = p(x)$. Then we have:

\begin{equation}
    \log p(x) = \mathbb{E}\left[\log R\right] + \mathbb{E}\left[\log p(x) - \log R\right]
\end{equation}

Using Jensen's Inequality, we can interpret the first term on the right as a lower bound on $\log p(x)$, and the second term as the bias. If we assume that $R$ stays relatively close to $p(x)$, then we can expand $\log R$ in a Taylor Series around $p(x)$:

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

This suggests we should look for a function of $R$ with the same mean but lower variance, one easy possibility is just the sample mean $R_K = \frac{1}{K}\sum_{k=1}^K R_k$. Then, as each $R_k$ has identical mean $p(x)$, we still have $\log p(x) \geq \E{}{\log R_K}$, but now this gives a tighter bound by a factor of $1/K$, with the caveat that $\vert \frac{R-p(x)}{p(x)} \vert < 1$. So using more importance samples helps us pump those rookie numbers up.

### **1.2. The Importance-Weighted ELBO**
We can connect this back to variational inference by letting $R$ have the following form:

\begin{equation}
    R = \frac{p(x,z)}{q(z)}, \quad z \sim q
\end{equation}

This satisfies the condition $\E{q}{R} = \int_Z dz\; q(z) \frac{p(x,z)}{q(z)} = p(x)$. By Jensen's Inequality, we have the following lower bound on the log marginal likelihood:

\begin{equation}
    \E{q}{\log R} \leq \log p(x)
\end{equation}

From our heuristic argument above, we can tighten this bound by sampling multiple $z_k \sim q$ and using the sample mean over $R$ to estimate $p(x)$, arriving at the importance-weighted ELBO, or the IW-ELBO [[1]](#1). Being a tighter variational bound, this is a useful but biased replacement for $\log p(x)$ for density estimation problems.

$$\begin{align}
    % R_K &= \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q \\
    \textrm{IW-ELBO}_K(x) &= \log \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q
\end{align}$$

Our argument makes it fairly intuitive that the IW-ELBO _should_ be a consistent estimator, but we probably need to tread more carefully - i.e. treat convergence of the Maclaurin series $\log(1+x)$ more carefully to demonstrate this formally.

### **1.3. IWAE in `Jax`**
Here's a barebones implementation of learning this bound with an amortized diagonal-Gaussian encoder/decoder pair in Jax, more details to follow in the next post. Note that easy vectorization in Jax makes this an almost trivial extension of the single importance sample VAE case.


{% highlight python %}
import jax as jnp
from jax import jit, vmap, random
from jax.scipy.special import logsumexp

def diag_gaussian_logpdf(x, mean=None, logvar=None):
    # Log of PDF for diagonal Gaussian for a single point
    D = x.shape[0]
    if (mean == None) and (logvar == None):
        # Standard Gaussian
        mean, logvar = jnp.zeros_like(x), jnp.zeros_like(x)
    jnp.sum(-0.5 * (jnp.log(2*jnp.pi) + logvar + (x-mean)**2 * jnp.exp(-logvar)))

def iw_estimator(x, rng, enc_params, dec_params):

    # Sample from q(z|x) by passing data through encoder and reparameterizing
    # Sample from p(x|z) by sampling from q(z|x), passing through decoder
    z_sample, x_sample, qzCx_stats, pxCz_stats = generate_samples(...)
    log_pxCz = diag_gaussian_logpdf(x, *pxCz_stats)
    log_qz = diag_gaussian_logpdf(z_sample, *qzCx_stats)
    log_pz = diag_gaussian_logpdf(z_sample)
    iw_log_summand = log_pxCz + log_pz - log_qz
    
    return iw_log_summand

@partial(jit, static_argnums=(5,))
def iwelbo_amortized(x, rng, enc_params, dec_params, num_samples=32, *args, **kwargs):

    rngs = random.split(rng, num_samples)
    vec_iw_estimator = vmap(iw_estimator, in_axes=(None, 0, None, None))
    iw_log_summand = vec_iw_estimator(x, rngs, enc_params, dec_params)

    K = num_samples
    iwelbo_K = logsumexp(iw_log_summand) - jnp.log(K)
    return iwelbo_K
{% endhighlight %}

## References

<a id="1">[1]</a> 
Burda, Yuri and Grosse, Roger and Salakhutdinov, Ruslan.
Importance Weighted Autoencoders.
International Conference on Learning Representations (2016).