---
layout: post
title: "Neural Image Compression, Part I"
date: 2020-09-12
categories: machine-learning, image-compression, pytorch
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

This post is the first in a multi-part series about learnable data compression.<!--more--> This post was developed when porting the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/) [[1]](#1) to PyTorch - you can find the [implementation on Github here](https://github.com/Justin-Tan/high-fidelity-generative-compression). The repository also includes general routines for lossless data compression which interface with PyTorch for all your data compression needs.

The bulk of the neural compression literature is focused on image/video compression, as we have well-developed prior models that allow us to extract the most relevant features for compression from the original input space in the case of images. However, the concept straightforwardly extends to other types of data. In this first part we'll discuss methods for lossy compression and an interesting connection to variational inference.

* Contents
{:toc}

## 1. Lossy Compression

Machine learning methods for lossy image compression encode an image $$\*x \in \mathcal{X} \subset \mathbb{R}^D$$, represented as a matrix of pixel intensities, into a latent representation $$\*z = E_{\phi}(\*x)$$ via the _encoder_ $E_{\phi}: \mathbb{R}^D \rightarrow \mathbb{R}^C$. This latent representation is subsequently quantized, $$\hat{\*z} = Q(\*z)$$, e.g. by rounding to the nearest integer. This reduces the amount of information required to transmit/store $$\*x$$, but introduces error into the representation.

The discrete form $$\hat{\*z}$$ can be losslessly stored using entropy coding methods and transmitted as a sequence of bits. Here a parametric model of the discrete marginal distribution $$p_{\nu}: \mathbb{Z}^C \rightarrow [0,1]$$ is introduced that aims to model the true marginal distribution $$p^*(\hat{\*z}) = \int_{\mathcal{X}} d\*x \; p(\*x, \hat{\*z})$$. The expected code length (or bitrate) achievable is given by the cross-entropy between the true and parametric marginals:

$$\begin{align}
R &= -\E{p^*}{\log_2 p_{\nu}(\hat{\*z})} \\
&= -\E{p^*}{\log_2 p^*(\hat{\*z})}+ \kl{p^*}{p_{\nu}}.
\end{align}$$

Owing to the positivity of the KL-divergence, $R$ is minimized if $$p^* = p_{\nu}$$, in agreement with Shannon's source coding theorem for memoryless sources. A generative _decoder_ model $G_{\theta}: \mathbb{R}^C \rightarrow \mathbb{R}^D$ can be used to recover a lossy reconstruction, $$\*x' = G_{\theta}(\hat{\*z}) \in \mathcal{X}'$$. The quality of this reconstruction is quantified using a _distortion function_:

$$ d: \mathcal{X} \times \mathcal{X}' \rightarrow \mathbb{R}^+. $$

Where $$d(\*x, \*x')$$ measures the cost of representing input $$\*x$$ with its reconstruction $$\*x'$$. For images, the mean squared error is a popular distortion measure. There exists a tradeoff between the expected code length of the compressed representation and the distortion. A higher rate results in a lower distortion, and vice-versa. This tradeoff can be explicitly manifested in a scalar hyperparameter $\beta$, and the parameters $(\theta, \phi, \nu)$ of the encoder, decoder and marginal model, respectively, are optimized jointly through minimization of the rate-distortion objective:

$$ \mathcal{L} = \E{p(\*x)}{d(\*x,\*x') + \beta R} = \E{p(\*x)}{d(\*x,G_{\theta} \circ Q \circ E_{\phi}(\*x)) + \beta R}.$$

### 1.1. Differentiable proxy for quantization

Naively applying gradient descent to this objective is problematic as the gradient w.r.t. $\phi$ will be zero almost everywhere due to quantization. As an alternative, one differentiable relaxation is to use additive random noise in lieu of hard quantization. Considering the single-dimensional case, the probability mass function from integer rounding is:

$$ p_{\nu}(\hat{z}) = \sum_{n \in \mathbb{Z}} w_n \delta(\hat{z} - n). $$

Where the weights 

$$w_n = \mathbb{P}\left(z \in \left(n-\frac{1}{2}, n+\frac{1}{2}\right)\right) = \int_{n-\frac{1}{2}}^{n+\frac{1}{2}} dz \; p_{\nu}^{\textrm{cont.}}(z) = p_{\nu}^{\textrm{cont.}}(n) * \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)$$

describe the probability mass at each quantization point $n \in \mathbb{Z}$. Using the fact that the density function of the sum of independent random variables is given by convolution of the respective densities, a continuous relaxation to the discrete distribution is obtained by passing $$z$$ through an additive unit-width uniform noise channel: $\hat{z} \approx z + u, \; u \sim \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)$. The resulting PDF matches the quantized PMF exactly at $n \in \mathbb{Z}$ and provides a continuous interpolation at intermediate values. For the multivariate case, assuming a fully-factored PMF $$p_{\nu}(\hat{\*z}) = \prod_i p_{\nu}(\hat{z}_i)$$, additive uniform noise drawn from the $(-1/2,1/2)^C$ hypercube is used in place of quantization:

$$\hat{\*z} \approx \*z + \*u, \quad \*u \sim \mathcal{U}\left(-\frac{1}{2},\frac{1}{2}\right)^C.$$

## 2. Bits-Back Coding

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

<!-- ## 3. Entropy Model
The final probability model is a two-level hierarchial model where the latents are described by a Gaussian or logistic distribution with spatially adaptive parameters. The latents $$\*y$$ are assumed conditionally independent given the hyperlatents $$\*z$$. The latter is modelled using a non-parametric factorized density with the logits are predicted for each subpixel. This setup reflects a prior belief that the likelihood of neighbouring values in the latent obtain should be position-dependent - e.g. neighbouring values in the latent domain should have similar probability, while allowing a large degree of freedom for the hyperlatent values. -->

## References

<a id="1">[1]</a> 
Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur.,
"High-Fidelity Generative Image Compression.",
arXiv:2006.09965 (2020).

<a id="2">[2]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston.,
"Variational image compression with a scale hyperprior",
arXiv:1802.01436 (2018).


<!-- ## Citation
Please cite the [original paper](https://arxiv.org/abs/2006.09965) if you use their work.
{% highlight python %}
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
{% endhighlight %} -->
