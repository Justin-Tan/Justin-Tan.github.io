---
layout: post
title: "Generative Image Compression"
date: 2020-07-28
categories: machine-learning, image-compression, pytorch, generative-models
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

This post talks about learnable data compression using machine learning methods.<!--more--> Here we focus on _generative compression_ methods which involve the use of a conditional GAN architecture to encourage realistic reconstructions. This post was created when attempting to implement the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/) [[1]](#1). Here we focus on image data but the concept straightforwardly extends to other types of data.

* Contents
{:toc}

## 1. Lossy Compression

Machine learning methods for lossy image compression encode an image $$\*x \in \mathcal{X} \subset \mathbb{R}^D$$, represented as a vector/matrix of pixel intensities, into a latent representation $$\*z = E_{\phi}(\*x)$$ via the _encoder_ $E_{\phi}: \mathbb{R}^D \rightarrow \mathbb{R}^C$. This latent representation is subsequently quantized, $$\*z \rightarrow \hat{\*z}$$, e.g. by rounding to the nearest integer. This reduces the amount of information required to transmit/store $$\*x$$, but introduces error into the representation. 

The discrete form $$\hat{\*z}$$ can be losslessly stored using entropy coding methods and transmitted as a sequence of bits. Here a parametric model of the discrete marginal distribution $$p_{\nu}: \mathbb{Z}^C \rightarrow [0,1]$$ is introduced that aims to model the true marginal distribution $$p^*(\hat{\*z}) = \int_{\mathcal{X}} d\*x \; p(\*x, \hat{\*z})$$. The expected code length (or bitrate) achievable is given by the cross-entropy between the true and parametric marginals:

$$\begin{align}
R &= -\E{p^*}{\log_2 p_{\nu}(\hat{\*z})} \\
&= -\E{p^*}{\log_2 p^*(\hat{\*z})}+ \kl{p^*}{p_{\nu}}.
\end{align}$$

Owing to the positivity of the KL-divergence, $R$ is minimized if $$p^* = p_{\nu}$$, in agreement with Shannon's source coding theorem. A generative _decoder_ model $G_{\theta}: \mathbb{R}^C \rightarrow \mathbb{R}^D$ can be used to recover a lossy reconstruction, $$\*x' = G_{\theta}(\hat{\*z}) \in \mathcal{X}'$$. The quality of this reconstruction is quantified using a _distortion function_:

$$ d: \mathcal{X} \times \mathcal{X}' \rightarrow \mathbb{R}^+. $$

Where $$d(\*x, \*x')$$ measures the cost of representing input $$\*x$$ with its reconstruction $$\*x'$$. For images, the mean squared error is a popular distortion measure. There exists a tradeoff between the expected code length of the compressed representation and the distortion. A higher rate results in a lower distortion, and vice-versa. This tradeoff can be explicitly manifested in a scalar hyperparameter $\beta$, and the parameters $(\theta, \phi, \nu)$ of the encoder, decoder and marginal model, respectively, are optimized jointly through minimization of the rate-distortion objective:

$$ \mathcal{L} = \E{p(\*x)}{d(\*x,\*x') + \lambda R} = \E{p(\*x)}{d(\*x,G_{\theta} \circ E_{\phi}(\*x)) + \beta R}.$$

Naively applying gradient descent to this objective is problematic as the gradient w.r.t. $\phi$ will be zero almost everywhere due to quantization. As an alternative, one differentiable relaxation is to use additive random noise in lieu of hard quantization. Considering the single-dimensional case, the probability mass function from integer rounding is:

$$ p_{\nu}(\hat{z}) = \sum_{n \in \mathbb{Z}} w_n \delta(\hat{z} - n). $$

Where the weights 

$$w_n = \mathbb{P}\left(z \in \left(n-\frac{1}{2}, n+\frac{1}{2}\right)\right) = \int_{n-\frac{1}{2}}^{n+\frac{1}{2}} dz \; p_{\nu}^{\textrm{cont.}}(z) = p_{\nu}^{\textrm{cont.}}(n) * \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)$$

describe the probability mass at each quantization point $n \in \mathbb{Z}$. Using the fact that the density function of the sum of independent random variables is given by convolution of the respective densities, the quantized form is obtained by the additive uniform noise channel $\hat{z} \approx z + u, \; u \sim \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)$. The resulting PDF matches the quantized PMF exactly at $n \in \mathbb{Z}$ and provides a continuous interpolation at intermediate values. For the multivariate case, assuming a fully-factored PMF $$p_{\nu}(\hat{\*z}) = \prod_i p_{\nu}(\hat{z}_i)$$, additive uniform noise drawn from the $(-1/2,1/2)^C$ hypercube is used in place of quantization:

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


## References

<a id="1">[1]</a> 
Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur.,
"High-Fidelity Generative Image Compression.",
arXiv:2006.09965 (2020).

<a id="2">[2]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston.,
"Variational image compression with a scale hyperprior",
arXiv:1802.01436 (2018).


## Citation
Please cite the [original paper](https://arxiv.org/abs/2006.09965) if you use their work.
{% highlight python %}
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
{% endhighlight %}
