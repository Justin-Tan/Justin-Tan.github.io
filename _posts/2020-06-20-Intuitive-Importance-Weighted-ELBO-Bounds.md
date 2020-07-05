---
layout: post
title: "Intuitive IWAE Bounds + Implementation in Jax"
subtitle: "Pumping those rookie numbers up."
date: 2020-06-20
categories: machine-learning, latent-variables, jax
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

Here we'll give a quick and dirty sketch about why importance-weighted autoencoders provide a lower bound on the log-marginal likelihood $\log p(x)$, then walk through a fast implementation in Jax.<!--more-->

* Contents
{:toc}

## 1. Importance Weighted Autoencoders
Recall that the setup is that we are attempting to model the log-marginal likelihood $\log p(x)$ over some given data $x$. The evidence lower bound (ELBO) lower bounds (surprise!) $\log p(x)$ with gap given by the KL divergence between the learnt approximation $q_{\lambda}(z)$ and the true posterior $p(z \vert x)$. We want to maximize the lower bound with respect to the variational parameters $\lambda$ of $q_{\lambda}$. For more information check out this [earlier primer post]({% post_url 2020-06-15-LVM-primer %}).

$$\begin{align}
\kl{q_{\lambda}(z)}{p(z \vert x)} &=  \E{q_{\lambda}(z)}{\log q_{\lambda}(z) -  \log  p(z \vert x)} \\
    &= \log p(x) + \underbrace{\E{q_{\lambda}(z)}{\log q_{\lambda}(z) - \log  p(x,z)}}_{\text{ELBO}(x)}
\end{align}$$

Of course the natural question is if we can reduce this gap, and the answer happens to be yes, without too much effort.

### 1.1 Heuristic Motivation
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

This suggests we should look for a function of $R$ with the same mean but lower variance, one easy possibility is just the sample mean $R_K = \frac{1}{K}\sum_{k=1}^K R_k$. Then, as each $R_k$ has identical mean $p(x)$, we still have $\log p(x) \geq \E{}{\log R_K}$, but now this gives a tighter bound by a factor of $1/K$, with the caveat that $\vert \frac{R-p(x)}{p(x)} \vert < 1$. So using more importance samples $K$ helps us pump those rookie numbers up.

### 1.2. The Importance-Weighted ELBO
We can connect this back to variational inference by letting $R$ have the following form:

\begin{equation}
    R = \frac{p(x,z)}{q(z)}, \quad z \sim q
\end{equation}

This satisfies the condition $\E{q}{R} = \int_Z dz\; q(z) \frac{p(x,z)}{q(z)} = p(x)$. By Jensen's Inequality, we have the following lower bound on the log marginal likelihood:

\begin{equation}
    \E{q}{\log R} \leq \log p(x)
\end{equation}

From our heuristic argument above, we can tighten this bound by sampling multiple $z_k \sim q$ and using the sample mean over $R$ to estimate $p(x)$, arriving at the importance-weighted ELBO, or the IW-ELBO, first introduced by Burda et. al. [[1]](#1). Being a tighter variational bound, this is a useful but biased replacement for $\log p(x)$ for density estimation problems.

$$\begin{align}
    % R_K &= \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q \\
    \textrm{IW-ELBO}_K(x) &= \log \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q
\end{align}$$

Our argument makes it fairly intuitive that the IW-ELBO _should_ be a consistent estimator with convergence rate linear in $K$, but we  need to tread more carefully to properly understand the asymptotics, by showing the remainder term in the Taylor expansion goes to zero as $K \rightarrow \infty$. Rainforth et. al. [[2]](#2) and Domke and Sheldon [[3]](#3) provide rigorous bounds on the asymptotic error term.

### 1.3. IWAE in `Jax`
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
    
 def diag_gaussian_sample(rng, mean, logvar):
    # Sample single point from a diagonal multivariate distribution
    return mean + jnp.exp(0.5 * logvar) * random.normal(rng, mean.shape)
{% endhighlight %}

First some convenience functions to evaluate the log-density of a diagonal Gaussian and to sample from a diagonal Gaussian using reparameterization. Note in Jax we explicitly pass a PRNG state, represented above by `rng` into any function where randomness is required, such as when sampling $\mathcal{N}(0,\mathbb{I})$ for reparameterization. The short reason for this is because standard PRNG silently updates the state used to generate pseudo-randomness, resulting in a hidden 'side effect' that is problematic for Jax's transformation and compilation functions, which only work on Python functions which are functionally pure. Instead Jax explicitly passes the PRNG state as an argument and splits the state as required whenever we require more instances of PRNG. Check out the [documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers) for more details. Next we define the log of the summand of the IW-ELBO estimator. Assume we have defined an amortized encoder/decoder that allows us to obtain the distribution parameters by passing each datapoint $x$ through their respective functions.

{% highlight python %}
def iw_estimator(x, rng, enc_params, dec_params):

    # Sample from q(z|x) by passing data through encoder and reparameterizing
    z_mean, z_logvar = encoder(x, enc_params)
    qzCx_stats = (z_mean, z_logvar)
    z_sample = diag_gaussian_sample(rng, *qzCx_stats)
    # Sample from p(x|z) by sampling from q(z|x), passing through decoder
    x_mean, x_logvar = decoder(z_sample, dec_params)
    pxCz_stats = (x_mean, x_logvar)
    
    log_pxCz = diag_gaussian_logpdf(x, *pxCz_stats)
    log_qz = diag_gaussian_logpdf(z_sample, *qzCx_stats)
    log_pz = diag_gaussian_logpdf(z_sample)
    
    iw_log_summand = log_pxCz + log_pz - log_qz
    
    return iw_log_summand
{% endhighlight %}

Apart from the `rng`, everything we have written down looks like plausible `numpy`/`torch` so far. The manner in which we collect the summands for different samples $z_k \sim q(z \vert x)$ is the only significant point of departure:

{% highlight python %}
def iwelbo_amortized(x, rng, enc_params, dec_params, num_samples=32, *args, **kwargs):

    rngs = random.split(rng, num_samples)
    vec_iw_estimator = vmap(iw_estimator, in_axes=(None, 0, None, None))
    iw_log_summand = vec_iw_estimator(x, rngs, enc_params, dec_params)

    K = num_samples
    iwelbo_K = logsumexp(iw_log_summand) - jnp.log(K)
    return iwelbo_K
{% endhighlight %}

First, we split the `rng` states into `num_samples` new states, one for each importance sample. Next we make use of [`jax.vmap`](https://jax.readthedocs.io/en/latest/jax.html#jax.vmap) - one of the 'killer features' of Jax. This function vectorizes a function over a given axis of the input, so the function is evaluated in parallel across the given axis. A trivial example of this would be representing a matrix-vector operation as a vectorized form of dot products between each row of the matrix and the given vector, but `vmap` allows us to (in most cases) vectorize more complicated functions where it is not obvious how to manually 'batch' the computation. `vmap` takes in three arguments:

- `fun`: This is the function to vectorize, in this case the function which computes the log of the summand for the IW-ELBO, we want to compute this in parallel across all the importance samples.
- `in_axes`: This is a tuple/integer specifying which axes of the input the function should be parallelized with respect to. In this case, our function has multiple arguments, so `in_axes` is a list/tuple of length given by the number of arguments to `fun`. Each element of `in_axes` represents the array axis to map over for the corresponding argument. A `None` means to not parallelize over this argument. Here the arguments of `iw_estimator` are `(x, rng, enc_params, dec_params)` - the input, PRNG state, and amortized encoder/decoder parameters, respectively. The input and the amortized parameters are constant for all importance samples, so the only argument we want the function to be parallelized over is the PRNG state. As `rngs` is a one-dimensional array, we want to parallelize over the first (`0`th) dimension, so `in_axes = (None, 0, None, None)`.'
- `out_axes`: Similar definition to `in_axes`, but specifying which axis of the function output the vectorized result should appear. This is usually `0` (the default) in most cases, following the standard convention of letting the first axis of an array represent the number of batch elements.

The return result is the vectorized version of `iw_estimator`, which we call to get a vector, `iw_log_summand`, representing the log-summands $\log p(x \vert z_k) + \log p(z_k) - \log q(z_k \vert x)$ for each importance sample $z_k$. This will be a one-dimensional array with number of elements given by the number of importance samples $K$. Finally for numerical stability we take the $\text{logsumexp}$ of the log-summands and average this to give the final IW-ELBO(K). Note how easy it was to automatically parallelize the computation of the different importance sample summands using `jax.vmap` - there's a Pytorch implementation in the Appendix that requires us to manually reason about the batch dimension during reparameterization, and involves a lot of wrangling over axes and dimensions in order to ensure we are performing a matrix-vector operation on the GPU and not a sequence of vector-vector operations.

## Conclusion

We are done, the IW-ELBO can be used as a drop-in replacement for the standard ELBO, trading off tightness of the bound for compute time. We didn't talk about two very important features of Jax - `jax.jit` and `jax.grad`, but we'll cover this more thoroughly in the next post, about other nice ways of estimating $\log p(x)$ and optimizing our estimates in terms of the amortized parameters. If you have any suggestions about the content or how the presented code can be optimized, please let me know!

## References

<a id="1">[1]</a> 
Burda, Yuri and Grosse, Roger and Salakhutdinov, Ruslan.
Importance Weighted Autoencoders.
International Conference on Learning Representations (2016).

<a id="2">[2]</a> 
Rainforth et. al.
Tighter Variational Bounds are not Necessarily Better.
International Conference on Machine Learning (2018).

<a id="3">[3]</a> 
Domke, Justin and Sheldon, Daniel.
Importance Weighting and Variational Inference.
NIPS (2018).

## Appendix: Pytorch Implementation

We omit the standard helper functions below for brevity. Imagine we had some standard `VAE` class with all the base functionality. There may be a more efficient way to do this in Torch rather than brute-force duplication along the batch axis, but it is reasonably fast with no significant slowdown for up to 256 importance samples using a batch size of 512 and input data with dimension 64. Note that, unlike Jax, a lot of the code is concerned with manually batching the importance samples into a single matrix for efficient vectorization. 

{% highlight python %}
class IWAE(VAE):
    def __init__(self, input_dim, hidden_dim, latent_dim=8, num_i_samples=32):
        
        super(IWAE, self).__init__(input_dim, hidden_dim, latent_dim)
        self.num_i_samples = num_i_samples


    def log_px_estimate(self, x, pxCz_stats, z_sample, qzCx_stats):

        x = torch.repeat_interleave(x, self.num_i_samples, dim=0)
        
        # [n*B,1]
        log_pz = math.log_density_gaussian(z_sample).sum(1, keepdim=True)
        log_qzCx = math.log_density_gaussian(z_sample, *qzCx_stats).sum(1, keepdim=True)
        log_pxCz = math.log_density_gaussian(x, mu=pxCz_stats['mu'], 
                    logvar=pxCz_stats['logvar']).sum(1, keepdim=True) 

        log_iw = log_pxCz + log_pz - log_qzCx
        log_iw = log_iw.reshape(-1, self.num_i_samples)  # [B,n]

        iwelbo = torch.logsumexp(log_iw, dim=1) - np.log(self.num_i_samples)
        return iwelbo

    def forward(self, x, **kwargs):
        """
        Tighten lower bound by reducing variance of marginal likelihood
        estimator through the sample mean.
        """

        qzCx_stats = self.encoder(x)
        mu_z, logvar_z = qzCx_stats  # [B,D]
        
        # Repeat num_i_samples times along batch dimension - [n,B,D]
        mu_z = torch.repeat_interleave(mu_z, num_i_samples, dim=0)
        logvar_z = torch.repeat_interleave(logvar_z, num_i_samples, dim=0)
        qzCx_stats = (mu_z, logvar_z)

        # Sample num_i_samples for each batch element
        z_sample = self.reparameterize_continuous(mu=mu_z, logvar=logvar_z)
        pxCz_stats = self.decoder(z_sample)

        return pxCz_stats, z_sample, qzCx_stats
{% endhighlight %}
   
  
