---
layout: post
title: "Monte Carlo Methods and Normalizing Flows"
date: 2020-07-30
categories: machine-learning, normalizing-flows, monte-carlo, jax
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

In this post we'll look at how normalizing flow models can be applied to Monte Carlo integration, with a fast example in Jax.<!--more--> The first two sections deal with basics about Monte Carlo and importance sampling, and the third section explores how flow models can be used to define a low-variance importance sampling density. The last section looks at how some difficult integrals can be approximated in Jax.

* Contents
{:toc}

## 1. Monte Carlo

Monte Carlo (MC) integration aims to approximate an expectation by the mean of a randomly drawn sample of random variables. Let $\*x \in \mathcal{X} \subseteq \mathbb{R}^D$ be some random variable with distribution $p_X(\*x)$. Assuming we are working in the continuous domain, the expectation of some function, $f: \mathcal{X} \rightarrow \mathbb{R}$ is:

$$\begin{equation}
\E{p}{f(\*x)} = \int_{\mathcal{X} dx \, p_X(\*x) f(\*x)}.
\end{equation}$$

If we take an i.i.d. sample $(\*x_1, \ldots, \*x_N)$ drawn from $p_X(\*x)$, then we define the _Monte Carlo estimate_ of the expectation as the sample mean (note this is an unbiased estimator):

$$\begin{equation}
\bar{f}_N(\*x) = \frac{1}{N}\sum_{n=1}^N f(x_n).
\end{equation}$$

The weak law of large numbers says that the MC estimate becomes arbitarily close to the true value, within some $\epsilon > 0$, as we increase the number of samples $N$:

$$\begin{equation}
\lim_{N \rightarrow \infty} \mathbb{P}\left( \vert \bar{f}_N(\*x) - \E{p}{f(\*x)} \vert  \geq \epsilon\right) = 0.
\end{equation}$$

The variance of the MC estimate scales as $1/N$, irrespective of the number of dimensions:

$$\begin{equation}
\mathbb{V}_p\left[\bar{f}(\*x)\right] = \frac{1}{N}\mathbb{V}_p\left[ f(\*x) \right] = \frac{1}{N} \int{\mathcal{X}} dx \, p_X(\*x) \left( f(\*x) - \E{p}{f(\*x)}\right)^2
\end{equation}$$

Integration problems can be naively converted to Monte Carlo problems by specifying an integration domain $\Omega$ and writing the integral as an expectation under the uniform distribution over $\Omega$:

$$ \int_{\Omega} dx, f(\*x) = \vert \Omega \vert \E{U(\Omega)}{f(\*x)} \approx \frac{\vert \Omega \vert}{N} \sum_{n} f(\*x_n), \quad \*x_n \sim U(\Omega) $$.

This is inefficient for high dimensional problems as the integrand $f(\*x)$ may only assume non-negligble values in special regions of $\Omega$.

## 2. Importance Sampling
Importance sampling is one way of reducing the variance of the MC estimate $\bar{f}$. The idea is to use a distribution other than $p(\*x)$ from which to generate the points $(\*x_1, \ldots, \*x_N)$ from. In fact, the integral we are interested in, $\int_{\mathcal{X}} dx \, f(\*x) p(\*x)$ can be represented in an infinite number of ways by the triplet $(f,p,\mathcal{X})$. One sensible way to proceed is to focus on those low-variance choices of triples. 
### 2.1. Example
Take the following example, from Robert and Casella [[2]](#2). Here we are interested in finding the exceedance $\geq 2$ of the Cauchy distribution $\mathcal{C}(0,1)$.

$$ p = \int_2^{\infty} \frac{dx}{\pi (1+x^2)}$$.

Naively, we can sample $x_n$ i.i.d. from $\mathcal{C}(0,1)$ and use the MC estimator:

$$ \hat{p}_1 = \frac{1}{N} \sum_n \mathbb{I}\left[x_n \geq 2\right]; \quad x_n \sim \mathcal{C}(0,1)$$

The summand is a Bernoulli r.v. so the variance of this estimator is $\mathbb{V} = p(1-p)/N \approx 0.127/N$. Another representation of $p$ is obtained through change of variables $x \rightarrow y^{-1}$.

$$ p = \int_0^{1/2} dy \, \frac{y^{-2}}{\pi (1 + y^{-2})} = \frac{1}{2} \E{\mathcal{U}(0,\frac{1}{2})}{\frac{y^{-2}}{\pi (1 + y^{-2})}}$$.

In this form, $p$ is the expectation over the uniform distribution over $(0,1/2)$ and the MC estimator becomes:

$$ \hat{p}_2 = \frac{1}{N} \sum_n \frac{y_n^{-2}}{\pi(1+y_n^{-2})}; \quad y_n \sim \mathcal{U}(0,1/2) = \frac{1}{N} \sum_n h(y_n)$$.

Integration by parts shows that the variance $\mathbb{V} = \frac{1}{N} 9.5 \times 10^{-5}$. So evaluating expectations based off simulated r.vs from the original distribution may be inefficient - particularly for heavy-tailed distributions, due to simulated values being generated outside the relevant domain - $(2,\infty)$ in this case.

### 2.2. Estimator
Importance sampling decouples the expectation from the distribution $p$, introducing a _proposal distribution_ $q$, that is easy to sample and evaluate, and satisfies $\textsf{supp}(q) \superset \textsf{supp}(p)$:

$$ \E{p}{f(\*x)} = \int_{\mathcal{X}} dx \, f(\*x) \frac{p(\*x)}{q(\*x)} q(\*x) = \E{q}{\frac{f(\*x)p(\*x)}{q(\*x)}} $$. The core idea is to choose the proposal $q$ to avoid drawing samples that lie outside the region where the integrand is non-negligible. 

The importance sampling estimator converges almost surely to the expectation of interest provided the condition on their respective supports is satisified. However the variance will depend strongly on the choice of proposal:

$$ \mathbb{V}_q\left[\frac{f(\*x)p(\*x)}{q(\*x)}\right] = \E{q}{\left(\frac{f(\*x)p(\*x)}{q(\*x)}\right)^2} - \left( \E{q}{\frac{f(\*x)p(\*x)}{q(\*x)}} \right)^2$$.

Immediately we see that the ratio $p/q$ should not be too large to avoid the estimator from receiving large shocks. A formal result for the minimum-variance proposal is as follows, noting that the second term in the expectation is independent of $q(\*x)$:

$$\begin{equation}
    \argmin_q \mathbb{V}_q\left[\frac{f(\*x)p(\*x)}{q(\*x)}\right] = \argmin_q \E{q}{\left(\frac{f(\*x)p(\*x)}{q(\*x)}\right)^2}
\end{equation}$$

So some informal desiderata for the proposal $q(\*x)$ are that:

- $q(\*x)$ should have larger support than $f(\*x)p(\*x)$, that is $q(\*x) > 0 \; \forall \{\*x \vert f(\*x)p(\*x) \neq 0\}$. To avoid issues in tail regions, $q$ should be heavy-tailed.
- $q$ should be approximately proportional to the absolute value of the integrand $\vert p(\*x)f(\*x) \vert$.
- $q$ supports efficient sampling and evaluation.

### 2.3. Minimizing Variance
There is a long history of constructing proposals chosen from some parametric family minimizing some divergence measure between $q$ and the minimum variance estimator $q^*$. One popular choice is the heavy-tailed mixture of Student's-$t$ distributions $\sum_k \pi_k t_{\nu_k}\left(\*x \vert \mathbf{\mu}_k, \mathbf{\Sigma}_k)\right)$, where the learnable parameters are $\left(\{\nu_k, \mathbf{\mu}_k, \mathbf{\Sigma}_k\}\right)$. Fortunately, there are more expressive methods available; the vogueness of generative modelling means that we know how to construct complicated distributions as a parametric transformation of a simpler base distribution. 

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

We can interpret the first term on the right as a lower bound on $\log p(x)$, and the second term as the bias. If we assume that $R$ stays relatively close to $p(x)$, then we can expand $\log R$ in a Taylor Series around $p(x)$:

$$\begin{align}
    \E{}{\log R} &= \E{}{\log\left(p(x) + R - p(x)\right)} \\
    &= \E{}{\log\left(p(x)\left(1+\frac{R - p(x)}{p(x)}\right)\right)} \\
    &\approx \E{}{\log p(x) + \frac{1}{p(x)}\left(R - p(x)\right) - \frac{1}{2p(x)^2} \left(R - p(x)\right)^2} \\
    &= \log p(x) - \frac{1}{2p(x)^2} \mathbb{V}\left[R\right]
\end{align}$$

So the 'slackness' in the bound (approximately) scales with the variance of $R$, $\mathbb{V}\left[R\right]$:

\begin{align}
    \log p(x) - \mathbb{E}\left[\log R\right] \approx \frac{1}{2p(x)^2}\mathbb{V}\left[R\right] \geq 0
\end{align}

Note to make use of this expansion, we should have that $R$ is concentrated near $p(x)$ in the sense that the Taylor series for $\log\left(1 + \frac{R-p(x)}{p(x)}\right)$ converges.

The scaling of bound tightness with variance suggests we should look for a function of $R$ with the same mean but lower variance, one easy possibility is just the sample mean $R_K = \frac{1}{K}\sum_{k=1}^K R_k$. Then, as each $R_k$ has identical mean $p(x)$, we still have $\log p(x) \geq \E{}{\log R_K}$, but now this gives a tighter bound by a factor of $1/K$, with the caveat that $\vert \frac{R-p(x)}{p(x)} \vert < 1$. So using more importance samples $K$ helps us pump those rookie numbers up.

### 1.2. The Importance-Weighted ELBO
We can connect this back to variational inference by letting $R$ have the following form:

\begin{equation}
    R = \frac{p(x,z)}{q(z)}, \quad z \sim q
\end{equation}

Where $q$ is some proposal distribution that can be efficiently evaluated + sampled from. This satisfies the condition $\E{q}{R} = \int_Z dz\; q(z) \frac{p(x,z)}{q(z)} = p(x)$. By Jensen's Inequality, we have the following lower bound on the log marginal likelihood:

\begin{equation}
    \E{q}{\log R} \leq \log p(x)
\end{equation}

From our heuristic argument above, we can tighten this bound by sampling multiple $z_k \sim q$ and using the sample mean over $R$ to estimate $p(x)$, arriving at the importance-weighted ELBO, or the IW-ELBO, first introduced by Burda et. al. [[1]](#1). Being a tighter variational bound than the regular ELBO, which corresponds to the single sample $K=1$ case, this is a useful (but still biased) proxy for $\log p(x)$ in problems involving density estimation.

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

First some convenience functions to evaluate the log-density of a diagonal Gaussian and to sample from a diagonal Gaussian using reparameterization. Note in Jax we explicitly pass a PRNG state, represented above by `rng` into any function where randomness is required, such as when sampling $\mathcal{N}(0,\mathbb{I})$ for reparameterization. The short reason for this is because standard PRNG silently updates the state used to generate pseudo-randomness, resulting in a hidden 'side effect' that is problematic for Jax's transformation and compilation functions, which only work on Python functions which are functionally pure. Instead Jax explicitly passes the PRNG state as an argument and splits the state as required whenever we require more instances of PRNG. Check out the [documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers) for more details. 

Next we define the log of the summand of the IW-ELBO estimator. Assume we have defined an amortized encoder/decoder that allows us to obtain the distribution parameters by passing each datapoint $x$ through their respective functions.

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

The return result is the vectorized version of `iw_estimator`, which we call to get a vector, `iw_log_summand`, representing the log-summands $\log p(x \vert z_k) + \log p(z_k) - \log q(z_k \vert x)$ for each importance sample $z_k$. This will be a one-dimensional array with number of elements given by the number of importance samples $K$. Finally for numerical stability we take the $\text{logsumexp}$ of the log-summands and average this to give the final IW-ELBO(K).

Note how easy it was to automatically parallelize the computation of the different importance sample summands using `jax.vmap`. There's a Pytorch implementation in the Appendix that requires us to manually reason about the batch dimension during reparameterization, and involves some wrangling over axes and dimensions in order to ensure we are performing a matrix-vector operation on the GPU and not a sequence of vector-vector operations. This approach requires you to roll your own ad-hoc manual batching system for each problem you encounter - moreover, when playing these sorts of games, you can get badly burned if you aren't reshaping things consistently/properly,

## Conclusion

We are done, the IW-ELBO can be used as a drop-in replacement for the standard ELBO, trading off tightness of the bound for compute time. We didn't talk about two very important features of Jax - `jax.jit` and `jax.grad`, but we'll cover this more thoroughly in the next post, about other nice ways of estimating $\log p(x)$ and optimizing our estimates in terms of the amortized parameters. If you have any suggestions about the content or how the presented code can be optimized, please let me know!

## References

<a id="1">[1]</a> 
Burda, Yuri and Grosse, Roger and Salakhutdinov, Ruslan.
Importance Weighted Autoencoders.
International Conference on Learning Representations (2016).

<a id="2">[2]</a> 
Casella, George and Robert, Christian.
Monte Carlo Statistical Methods
Springer (2004).

[<a id="3">[3]</a> 
Domke, Justin and Sheldon, Daniel.
Importance Weighting and Variational Inference.
NIPS (2018).
]
## Appendix: Pytorch Implementation

We omit the standard helper functions below for brevity. Imagine we had some standard `VAE` class with all the base functionality. There may be a more efficient way to do this in Torch rather than brute-force duplication along the batch axis, but it is reasonably fast with no significant slowdown for up to 256 importance samples using a batch size of 512 and input data with dimension 64. Note that, unlike Jax, a lot of the code is concerned with manually batching the importance samples into a single matrix for efficient vectorization. A more elegant implementation may be possible using the excellent `torch.Distributions`, but it's nice to do things explicitly where possible.  

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
   
  
