---
layout: post
title: "Latent Variable Models in Jax"
subtitle: "For density estimation/sampling."
date: 2020-06-22
categories: machine-learning, latent-variables, jax
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

In this post we'll examine applications of LVMs to density estimation/approximate sampling. We'll implement the recently proposed SUMO (Stochastically Unbiased Marginalization Objective) in Jax, highlighting how Jax differs from standard Autodiff frameworks in some major aspects.<!--more-->  We end with the obligatory illustrative toy problem. If you are only interested in the practical part, skip to Section 2 or check out the [associated repository on Github](https://github.com/justin-tan/density_estimation_jax). This post did turn out longer than expected, so fair warning! There's a nice picture of my neighbour's cat at the end for some extra motivation.

* Contents
{:toc}

# 1. SUMO
The SUMO [[1]](#1) estimator makes use of the importance-weighted ELBO (IW-ELBO), examined in [an earlier post]({% post_url 2020-06-20-Intuitive-Importance-Weighted-ELBO-Bounds%}). Briefly, we are interested in finding a lower bound of the log-marginal likelihood $\log p(x)$. Under certain conditions, the tightness of this bound scales with the variance of some unbiased estimator $R$ of the marginal probability. $R$ depends on the latent variables through some proposal distribution $q(z)$: $\E{q}{R} = p(x)$. The regular VAE takes $R = p(x,z)/q(z)$, forming the normal ELBO $\E{q}{\log p(x,z) - q(z)}$, but a natural way to reduce the sample variance is through the $K$-sample mean, forming the IW-ELBO:

$$\begin{align}
    R_K &= \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q \\
    \textrm{IW-ELBO}_K(x) &= \log \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}, \quad z_k \sim q
\end{align}$$

One can show [[2]](#2) that the IW-ELBO is monotonically increasing in expectation and converges to the true log-marginal likelihood as the variance vanishes as $K \rightarrow \infty$: 

$$\begin{equation}
\E{q}{\textrm{IW-ELBO}_{K+1}(x)} \geq \E{q}{\textrm{IW-ELBO}_{K}(x)}, \quad \lim_{K \rightarrow \infty}\E{q}{\textrm{IW-ELBO}_{K}(x)} = \log p(x)
\end{equation}$$

So it appears we can only obtain an unbiased estimator of $\log p(x)$ after doing an infinite amount of computation.

## 1.1. Russian Roulette

SUMO circumvents this difficulty by randomized truncation of a specially constructed infinite series - falling under the class of the so-called 'Russian roulette estimators' [[3]](#3). Where exactly the specific form of the estimator comes from in the paper is a bit mysterious, so we'll motivate it heuristically below. For a general discussion, let $\Delta_k$ be the $k$th term of some series that converges to some quantity of interest $p^* = \lim_{K \rightarrow \infty} \left( \sum_{k=1}^K \Delta_k\right)$. We want to truncate the series at some finite stopping time $K$, while retaining unbiasedness. 

This is possible if we treat $K$ as a discrete random variable with mass function $p(K)$ with support over the natural numbers. The core idea is to draw $K \sim p(K)$, evaluate the first $K$ terms in the sum (each appropriately weighted to account for truncation), and stop. Taking the expectation of the partial sum w.r.t. $p(K)$ and shuffling things around in the resulting double sum[^1]:

$$\begin{align}
\E{p(K)}{\sum_{k=1}^K \Delta_k(x)} &= \sum_{K=1}^{\infty} p(K) \sum_{k=1}^K \Delta_k(x) \\
&= \sum_{K=1}^{\infty} \sum_{k=1}^{\infty} p(K)\Delta_k(x) \mathbb{I}(K \geq k)\\
&= \sum_{k=1}^{\infty} \Delta_k(x) \sum_{K=1}^{\infty} p(K)\mathbb{I}(K \geq k) \\
&= \sum_{k=1}^{\infty} \Delta_k(x) \mathbb{P}(K \geq k)
\end{align}$$

Where $\mathbb{P}(K \geq k)$ is the complement of the CDF. This suggests we can obtain an unbiased estimator of the infinite series by first sampling the stopping time $K$, and then evaluating $K$ appropriately weighted terms[^2]:

$$\begin{equation}
\sum_{k=1}^{\infty} \Delta_k(x) = \E{K \sim p(K)}{\sum_{k=1}^K \frac{\Delta_k(x)}{\mathbb{P}(K \geq k)}}
\end{equation}$$

## 1.2. Estimator 
To connect this back to lower bounds on $\log p(x)$, note that the following telescoping series converges to $\log p(x)$ in expectation in the limit:

$$\begin{equation}
\log p(x) = \E{q}{\textrm{IW-ELBO}_1(x) + \lim_{K \rightarrow \infty} \left[ \sum_{k=1}^K \left( \textrm{IW-ELBO}_{k+1}(x) - \textrm{IW-ELBO}_k(x)\right)\right]}
\end{equation}$$

Let $\hat{\Delta}\_k(x)= \E{q}{\textrm{IW-ELBO}_{k+1}(x)} - \E{q}{\textrm{IW-ELBO}_k(x)}$ then apply the Russian roulette estimator to the sequence of partial sums: 

$$\begin{align}
\log p(x) &= \E{q}{\textrm{IW-ELBO}_1(x)} + \sum_{k=1}^{\infty} \hat{\Delta}_k(x) \\
&= \E{q}{\textrm{IW-ELBO}_1(x)} +  \E{K \sim p(K)}{\sum_{k=1}^K \frac{\hat{\Delta}_k(x)}{\mathbb{P}(K \geq k)}} \\
&= \E{K \sim p(K)}{\E{q}{\textrm{IW-ELBO}_1(x) + \sum_{k=1}^K \frac{\textrm{IW-ELBO}_{k+1}(x) - \textrm{IW-ELBO}_k(x)}{\mathbb{P}(K \geq k)}}}
\end{align}$$

Luo et. al. [[1]](#1) define the interior of the expectation to be the Stochastically Unbiased Marginalization Objective (SUMO) estimator, which is equal to the marginal log-probability in expectation over $K$ and $q(z)$:

$$\begin{equation}
\textrm{SUMO}(x) = \textrm{IW-ELBO}_1(x) + \sum_{k=1}^K \frac{\textrm{IW-ELBO}_{k+1}(x) - \textrm{IW-ELBO}_k(x)}{\mathbb{P}(K \geq k)}, \quad z_k \sim q(z), K \sim p(K)
\end{equation}$$

Defining $\Delta_k(x)= \textrm{IW-ELBO}_{k+1}(x) - \textrm{IW-ELBO}_k(x)$, we arrive at the slightly more palatable form:

$$\begin{equation}
\textrm{SUMO}(x) = \log \frac{p(x,z_1)}{q(z_1)} + \sum_{k=1}^K \frac{\Delta_k(x)}{\mathbb{P}(K \geq k)}, \quad z_k \sim q(z), K \sim p(K)
\end{equation}$$

## 1.3. Variance Reduction
As Lyne et. al. [[3]](#3) (Appendix A.2.) demonstrate, the variance of the generic Russian roulette estimator can be potentially infinite - there is a tradeoff between computation time (controlled by the expected value of the stopping time and hence the form of the PMF $p(K)$), and the variance of the resulting estimator. Luo et. al. [[1]](#1) propose that the variance of SUMO can be minimized through optimization of the parameters $\phi$ of an amortized encoder $q_{\phi}(z \vert x)$, noting that:

$$\begin{align}
\nabla_{\phi}\mathbb{V}\left[\textrm{SUMO}(x)\right] &= \nabla_{\phi}\E{q_{\phi}, \, p(K)}{\textrm{SUMO}(x)^2} - \nabla_{\phi}\left(\E{q_{\phi}, \, p(K)}{\textrm{SUMO}(x)}\right)^2 \\
&= \nabla_{\phi}\E{q_{\phi}, \, p(K)}{\textrm{SUMO}(x)^2} \\
&= \E{p(\epsilon), \, p(K)}{\nabla_{\phi}\textrm{SUMO}(x)^2}
\end{align}$$

Where $\nabla_{\phi}\left(\E{q_{\phi}, \, p(K)}{\textrm{SUMO}(x)}\right)^2= \nabla_{\phi} \left(\log p(x)\right)^2 = 0$ by virtue of SUMO being an unbiased estimator of $\log p(x)$, and we apply reparameterization w.r.t. $z$ in the last line to pull the gradient into the expectation. So here the latent variables are doing double duty - used to obtain a sequence of lower bounds on $\log p(x)$, and to minimize the variance of the resulting estimator.

# 2. Implementation in Jax

[Jax is a machine learning framework](https://github.com/google/jax) described as the spiritual successor of [autograd](https://github.com/HIPS/autograd). Here's the elevator pitch: `numpy` + automatic differentiation, plus:

- Any-order gradients of your functions w.r.t. any input parameters through `jax.grad`.
- Automatic vectorization of single-example code into batches throguh `jax.vmap`.
- Automatic parallelization of code across multiple devices through `jax.pmap`.
- Automatic generation of [optimized computation kernels](https://www.tensorflow.org/xla) through `jax.jit`.

All the above primitives can be arbitrarily composed as well. From my experience, Jax was easy to come to grips with, sharing much of its minimalist interface with `numpy`/`scipy` and is blazingly fast out-of-the-box - switching from CPU to GPU/TPU 'just works', and it is straightforward to fuse your code into performant kernels or parallelize across multiple devices. 

However, Jax adopts a functional programming, as opposed to an object-oriented model, in order for function transformations to work nicely. This imposes certain constraints that may be unnatural to users accustomed to machine learning frameworks written in Python (Torch/Tensorflow). Python functions subject to transformation/compilation in Jax must be functionally pure: all the input data is passed through the function parameters, and all the results are output through the returned function values - there should be no dependence on global state or additional 'side-effects' beyond the return values. This can be somewhat vexing to those of us accustomed to calling `loss.backward()` in Torch - in fact, there's a [very long list of common gotchas in Jax](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) associated with this functionally pure constraint, but we'll see that this is not too big a barrier once you get used to it.

This section is quite long, but implementing and optimizing the SUMO estimator will force us to deal with a surprisingly large cross-section of Jax, together with many, many gotchas (at the time of writing this, as of v0.1.69). 

## 2.1. Defining the IW-ELBO

First we need a way to get the importance-weighted ELBO terms that appear in the Russian roulette estimator. This part is a lighter version of the full discussion in the [post about the importance-weighted ELBO]({% post_url 2020-06-20-Intuitive-Importance-Weighted-ELBO-Bounds%}). We'll start off with the most `numpy`-like part of the code, defining convenience functions to evaluate the log-density of a diagonal-covariance Gaussian and to sample from a diagonal Gaussian using reparameterization.

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

Note in Jax we explicitly pass a PRNG state, represented above by `rng` into any function where randomness is required, such as when sampling $\mathcal{N}(0,\mathbb{I})$ for reparameterization. The short reason for this is because standard PRNG silently updates the state used to generate pseudo-randomness, resulting in a hidden 'side effect' that is problematic for Jax's transformation and compilation functions, which only work on Python functions which are functionally pure. Instead Jax explicitly passes the PRNG state as an argument and splits the state as required whenever we require more instances of PRNG. Check out the [documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers) for more details. 

Here we are performing amortized inference - instead of learning variational parameters per-datapoint, we train separate neural networks to output the parameters of the approximate posterior $q(z \vert x)$ and conditional model $p(x \vert z)$. These are called the encoder/decoder, respectively. The variational parameters are then the collective parameters of these networks - shared between all datapoints. Using a standard Gaussian for the prior $p(z)$ and diagonal-covariance Gaussian distribution for the conditional and approximate posterior, the necessary samples to compute the IW-ELBO are obtained through reparameterization - see Section 3.2. of [this earlier post]({% post_url 2020-06-15-LVM-primer%}) for more details.

 $$\begin{align}
     (\mu_z, \Sigma_z) &= \textrm{Encoder}(x) \\
     z \sim q(z \vert x) \quad &\equiv \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{1}), \; z = \mu_z + \Sigma_z^{1/2} \epsilon  \\
     (\mu_x, \Sigma_x) &= \textrm{Decoder}(z) \\
     x \sim p(x \vert z) \quad &\equiv \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{1}), \; x = \mu_x + \Sigma_x^{1/2} \epsilon  \\
 \end{align}$$
 
Due to the functional programming paradigm, defining the encoder/decoder networks in `Jax` is slightly different from Torch/Tensorflow, etc. The core idea is that the constructor for each layer in the network expects an argument specifying the output dimension and returns:

- An `init_fun`, which initializes the layer parameters.
- An `apply_fun`, which defines the forward pass of the layer. 

The Jax-native `stax` library provides a straightforward way to compose layers into a single model, wrapping them together according to a user-defined structure, and returning an `init_fun` which initializes parameters for all layers in the model and an `apply_fun` which defines the forward pass of the model as the composition of the forward computations for each layer. See [the documentation](https://jax.readthedocs.io/en/latest/_modules/jax/experimental/stax.html#Dense) for more details. Here both the amortized encoder/decoder are defined as single-hidden-layer networks, where the output of each is split into two pieces - the mean and diagonal of the covariance matrix over the respective space.

{% highlight python %}
from jax.experimental import stax
def encoder_constructor(hidden_dim, z_dim, activation='Tanh'):
    activation = getattr(stax, activation)
    encoder_init, encoder_forward = stax.serial(
        stax.Dense(hidden_dim), activation,
        stax.FanOut(2),
        stax.parallel(stax.Dense(z_dim), stax.Dense(z_dim)),
    )
    return encoder_init, encoder_forward

def decoder_constructor(hidden_dim, x_dim=2, activation='Tanh'):
    activation = getattr(stax, activation)
    decoder_init, decoder_forward = stax.serial(
        stax.Dense(hidden_dim), activation,
        stax.FanOut(2),
        stax.parallel(stax.Dense(x_dim), stax.Dense(x_dim)),
    )
    return decoder_init, decoder_forward

encoder_init, encoder_forward = encoder_constructor(hidden_dim=hidden_dim, z_dim=z_dim)
decoder_init, decoder_forward = decoder_constructor(hidden_dim=hidden_dim, x_dim=x_dim)
{% endhighlight %}

The initialization functions accept:
- A PRNG state for random initialization of parameters
- A tuple specifying the dimensions of the network input.

They return:
- `output_shape`: A tuple specifying the dimensions of the network output.
- `init_params`: A tuple holding the initial network parameters.

{% highlight python %}
rng = random.PRNGKey(42)
rng, dec_init_rng, enc_init_rng = random.split(rng, 3)
z_output_shape, init_encoder_params = encoder_init(enc_init_rng, (-1, feature_dim))
x_output_shape, init_decoder_params = decoder_init(dec_init_rng, (-1, latent_dim))
{% endhighlight %}

Next we define the log of the summand of the IW-ELBO estimator. Recall the $K$-sample IW-ELBO is defined as:

$$\begin{align}
    \textrm{IW-ELBO}_K(x) &= \log \frac{1}{K} \sum_{k=1}^K \frac{p(x, z_k)}{q(z_k \vert x)}, \quad z_k \sim q
\end{align}$$

{% highlight python %}
def iw_estimator(x, rng, encoder, enc_params, decoder, dec_params):

    # Sample from q(z|x) by passing data through encoder and reparameterizing
    z_mean, z_logvar = encoder(enc_params. x)  # Encoder fwd pass
    qzCx_stats = (z_mean, z_logvar)
    z_sample = diag_gaussian_sample(rng, *qzCx_stats)
    # Sample from p(x|z) by sampling from q(z|x), passing through decoder
    x_mean, x_logvar = decoder(dec_params, z_sample)  # Decoder fwd pass
    pxCz_stats = (x_mean, x_logvar)
    
    log_pxCz = diag_gaussian_logpdf(x, *pxCz_stats)
    log_qz = diag_gaussian_logpdf(z_sample, *qzCx_stats)
    log_pz = diag_gaussian_logpdf(z_sample)
    
    iw_log_summand = log_pxCz + log_pz - log_qz
    
    return iw_log_summand
{% endhighlight %}


Note when applying the forward pass through the encoder and decoder, we have to explicitly pass the parameters of each network as arguments  - this is in line with Jax's functional programming interface - functions should only rely on their arguments, not global state. If you don't like this functional interface, the good news is that there are two projects - [flax](https://github.com/google/flax) and [trax](https://github.com/google/trax) which provide a more object-oriented interface for network construction (these are a lot more readable/hackable, IMO). 

The manner in which we collect the summands for different samples $z_k \sim q(z \vert x)$ represents another significant point of departure from other autodiff frameworks:

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

The return result is the vectorized version of `iw_estimator`, which we call to get a vector, `iw_log_summand`, representing the log-summands $\log p(x \vert z_k) + \log p(z_k) - \log q(z_k \vert x)$ for each importance sample $z_k$. This will be a one-dimensional array with number of elements given by the number of importance samples $K$. Finally for numerical stability we take the $\text{logsumexp}$ of the log-summands and average this to give the final IW-ELBO(K). Note how easy it was to automatically parallelize the computation of the different importance sample summands using `vmap`.


## 2.2. Defining the SUMO estimator

The authors propose the following tail distribution for the number of terms $\mathcal{K}$ to evaluate in the series: $\mathbb{P}(\mathcal{K} \geq k) = \frac{1}{k}$. This has CDF $F_{\mathcal{K}}(k) = \mathbb{P}(\mathcal{K} \leq k) = 1 - \frac{1}{k+1}$ and so by the inverse CDF transform $\left \lfloor{\frac{u}{1-u}}\right \rfloor \sim p(\mathcal{K})$, where $u \sim \text{Uniform}[0,1]$[^3]. 

# 3. Approximate Sampling
Here we would like to generate samples from the target distribution $p^*(x)$ using the model $p_{\theta}$ as a surrogate. This is achieved by sampling from the approximate posterior in latent space and subsequently sampling from the conditional model:

$$\begin{equation}
    x \sim p_{\theta}(x) \equiv z \sim q_{\lambda}(z), \quad x \sim p_{\theta}(x \vert z)
\end{equation}$$

If we are given some density $p^*(x) \propto \exp\left(-U(x)\right)$, where $U(x)$ is some energy function, the reverse-KL objective can be efficiently optimized for this purpose:

$$\begin{align}
    \mathcal{L}(\theta) = \kl{p_{\theta}(x)}{p^*(x)} &= \E{p_{\theta}(x)}{\log \frac{p_{\theta}(x)}{p^*(x)}} \\
    &= \E{p_{\theta}(x)}{\log p_{\theta}(x) - \log p^*(x)} \\
    &= -\mathbb{H}(p_{\theta}) + \E{p_{\theta}(x)}{U(x)} + \text{const.}
\end{align}$$

The target distribution $p^*$ in this case is defined as (originally proposed in [[4]](#4)):

$$ p^*(x_1,x_2) = \mathcal{N}(x_1 \vert 0, 1.35^2) \cdot \mathcal{N}\left(x_2 \vert 0, \exp(2 x_1)\right)$$

The distribution is shaped like a funnel, with a sharp neck in $x_2$ due to the exponential function applied to $x_1$. Note we can straightforwardly sample from $p^*$ by reparameterization: $x_1 = 1.35 \cdot \epsilon$, $x_2 = \exp(x_1) \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0,1)$, but we'll see how SUMO fares.

## 3.1. Objective Function

## 3.2. Optimization

## 3.3. Nice Plots


# Conclusion
Jax has some very interesting design choices. In particular, automatic vectorization through `vmap`{:.python} helps with mental overhead a lot once you don't have to carry around a cumbersome batch dimension. Personally, there's not enough of a QoL improvement at the moment over Torch for me to port my existing libraries to Jax, but I do enjoy tinkering with it and its minimalism lends itself well to quick proof of concept sketches (provided you are well-acquainted with the sharp edges). Next we might implement some interesting continuous? normalizing flow variants, which could be really easy, or troublesome, depending on the behaviour of `jax.jit`{:.python}.

[^1]: We are abandoning any pretense of formality here, ignoring all prerequisite conditions on $p(K)$ and $\Delta_k$ - see Appendix A.2. of [[3]](#3) if you have aversion to mathematical sin.

[^2]: I realize this doesn't look quite kosher, but do the same shuffling around of terms in the original derivation with the new estimator to convince yourself. 

[^3]: To see this, note $F_{\mathcal{K}}\left(F^{-1}\_{\mathcal{K}}(k)\right) = 1 - \frac{1}{F^{-1}\_{\mathcal{K}}(k)+1} = k \implies F^{-1}_{\mathcal{K}}(k) = \frac{k}{1-k}$, then take the floor to map to an integer.

## Credits
1. [Rob Lange's blog post](https://roberttlange.github.io/posts/2020/03/blog-post-10/) was very helpful when wrangling over the optimization process for Jax.
2. [Jax FAQ](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) helped out a lot when I was banging my head against my keyboard - Jax error traces can be a bit ... cryptic. 
3. [Bruce Springsteen](https://youtu.be/k1wqVIZR0ho?t=147) for providing the soundtrack for this post.

## References

<a id="1">[1]</a> 
Luo, Yucen, et. al.
SUMO: Unbiased Estimation of Log Marginal Probability for Latent Variable Models.
International Conference on Learning Representations (2020).
https://openreview.net/forum?id=SylkYeHtwr

<a id="2">[2]</a> 
Burda, Yuri and Grosse, Roger and Salakhutdinov, Ruslan.
Importance Weighted Autoencoders.
International Conference on Learning Representations (2016).

<a id="3">[3]</a> 
Lyne, Anne-Marie, et.al. 
On Russian Roulette Estimates for Bayesian Inference with Doubly-Intractable Likelihoods
Statistical Science. Institute of Mathematical Statistics, (2015).

<a id="4">[4]</a> 
Neal, Radford M.
Slice Sampling.
Annals of Statistics 31 (3): 705–67 (2003).


![Image](/assets/images/shell_web1.jpg)
_Cat Tax._