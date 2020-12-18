---
layout: post
title: "Neural Image Compression I"
date: 2020-09-12
categories: machine-learning, image-compression, pytorch
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
subtitle: Modern learnable data compression schemes use neural networks to define the transforms used in transform coding. We illustrate the basic idea behind learnable lossy compression and look at one possible continuous relaxation to quantization, required for entropy coding. 
---

This post is the first in a hopefully multi-part series about learnable data compression.<!--more--> This post was developed when porting the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/) [[1]](#1) to PyTorch - you can find the [resulting implementation on Github here](https://github.com/Justin-Tan/high-fidelity-generative-compression). The repository also includes general routines for lossless data compression which interface with PyTorch for all your compression needs.

The bulk of the neural compression literature is focused on image/video compression - we have well-developed prior models that allow us to extract the most relevant features for compression from the original input space. for image-based data. However, the same concept applies to other types of data, in principle. In this first part we'll discuss the rate-distortion objective used in transform coding and how to propagate gradients through the quantization operation for end-to-end learning.

* Contents
{:toc}

## Lossy Compression

Machine learning methods for lossy image compression encode an image $$\*x \in \mathcal{X} \subset \mathbb{R}^D$$, represented as a matrix of pixel intensities, into a latent representation $$\*z = E_{\phi}(\*x)$$ via the _encoder_ $E_{\phi}: \mathbb{R}^D \rightarrow \mathbb{R}^C$ (typically $C \ll D$). This latent representation is subsequently quantized, $$\hat{\*z} = \lfloor \*z \rceil$$, e.g. by rounding to the nearest integer. This reduces the amount of information required to transmit/store $$\*x$$, but introduces error into the representation.

The discrete form $$\hat{\*z}$$ can be losslessly compressed using standard entropy coding methods (e.g. Huffman, arithmetic, rANS) and transmitted as a sequence of bits. Here a parametric model of the discrete marginal distribution $$p_{\nu}: \mathbb{Z}^C \rightarrow [0,1]$$ is introduced that aims to model the true marginal distribution $$p^*(\hat{\*z}) = \int_{\mathcal{X}} d\*x \; p(\*x) \delta\left( \hat{\*z} - \lfloor E_{\phi}(\*x) \rceil\right)$$. The expected code length (or bitrate) achievable is given by the cross-entropy between the true and parametric marginals:

$$\begin{align}
R(\hat{\*z}) &= -\E{p^*}{\log_2 p_{\nu}(\hat{\*z})} \\
&= -\E{p^*}{\log_2 p^*(\hat{\*z})}+ \kl{p^*}{p_{\nu}}.
\end{align}$$

Owing to the positivity of the KL-divergence, $R$ is minimized if $$p^* = p_{\nu}$$, in agreement with Shannon's source coding theorem for memoryless sources.

The receiver can decode the bitstream using the same prior probability model the sender used to compress $$\hat{\*z}$$. A generative _decoder_ model $G_{\theta}: \mathbb{R}^C \rightarrow \mathbb{R}^D$ may then be used to recover a lossy reconstruction, $$\*x' = G_{\theta}(\hat{\*z}) \in \mathcal{X}'$$. The quality of this reconstruction is quantified using a _distortion function_:

$$ d: \mathcal{X} \times \mathcal{X}' \rightarrow \mathbb{R}^+. $$

Where $$d(\*x, \*x')$$ measures the cost of representing input $$\*x$$ with its reconstruction $$\*x'$$. For images, the mean squared error is a popular distortion measure. There exists a tradeoff between the expected code length of the compressed representation and the distortion. A higher rate results in a lower distortion, and vice-versa. This tradeoff can be explicitly manifested in the Lagrange multiplier $\beta$, and the parameters $(\theta, \phi, \nu)$ of the encoder, decoder and prior probability model, respectively, are optimized jointly in an end-to-end fashion via minimization of the rate-distortion objective:

$$\begin{align}
\mathcal{L}(\theta, \phi, \nu) &= \E{p(\*x)}{d(\*x,\*x') + \beta \cdot R(\hat{\*z})} \\
&= \underbrace{\E{p(\*x)}{d\left(\*x,G_{\theta} \circ  \lfloor E_{\phi}(\*x) \rceil\right)}}_{Distortion} + \beta \cdot \underbrace{\E{p(\*x)}{\left( -\E{p^*}{\log_2 p_{\nu}(\lfloor E_{\phi}(\*x) \rceil \right)}}}_{Rate} \\
&= \textrm{Expected reconstruction mismatch} + \beta \cdot \left( \textrm{Expected bitstream length}\right)
\end{align}$$

For image-based data, the encoder and generator transforms are usually parameterized using convolutional networks, which may be able to achieve a more appropriate latent representation compared to the linear transforms used by traditional image codecs. The distribution parameters of the prior probability model are also typically defined by a convolutional/dense network - e.g. if the probability model over $$\hat{\*z}$$ is taken to be Gaussian, the mean and scale parameters of the distribution are defined by the output of a neural network, as in Ballé et. al. [[2]](#2). We'll discuss this more in the next post.

## Differentiable quantization proxy

Naively applying gradient descent to this objective is problematic as the gradient w.r.t. $\phi$ and $\nu$ will be zero almost everywhere due to quantization. As an alternative, one differentiable relaxation is to use additive random noise in lieu of hard quantization. Considering the single-dimensional case, the probability mass function from integer rounding is:

$$ p_{\nu}(\hat{z}) = \sum_{n \in \mathbb{Z}} w_n \delta(\hat{z} - n). $$

Where the weights

$$w_n = \mathbb{P}\left(z \in \left(n-\frac{1}{2}, n+\frac{1}{2}\right)\right) = \int_{n-\frac{1}{2}}^{n+\frac{1}{2}} dz \; p_{\nu}^{\textrm{cont.}}(z) = p_{\nu}^{\textrm{cont.}}(n) * \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)$$

describe the probability mass at each quantization point $n \in \mathbb{Z}$. Using the fact that the density function of the sum of independent random variables is given by convolution of the respective densities, a continuous relaxation to the discrete distribution is obtained by passing $$z$$ through an additive unit-width uniform noise channel: $\hat{z} \approx z + u, \; u \sim \mathcal{U}\left(-\frac{1}{2}, \frac{1}{2}\right)$. The resulting PDF matches the quantized PMF exactly at $n \in \mathbb{Z}$ and provides a continuous interpolation at intermediate values. For the multivariate case, assuming a fully-factored PMF $$p_{\nu}(\hat{\*z}) = \prod_i p_{\nu}(\hat{z}_i)$$, additive uniform noise drawn from the $(-1/2,1/2)^C$ hypercube is used in place of quantization:

$$\hat{\*z} \approx \*z + \*u, \quad \*u \sim \mathcal{U}\left(-\frac{1}{2},\frac{1}{2}\right)^C.$$

Another alternative is to just use the 'straight-through estimator', where the gradients of the rounding operation are defined to be the identity: $$ \nabla \lfloor \*z \rceil \triangleq I$$ - in effect ignoring quantization during the backward pass. It appears that modern approaches, including the HiFIC paper [[1]](#1), use an ad-hoc mixed approach where the uniform noise channel is used to learn the prior probability model, but the straight-through estimator is used when $$\lfloor \*z \rceil $$ is passed as input to the generator. In either case, the end result is that the (approximated) gradients can flow freely backward, and the rate-distortion Lagrangian can be optimized using stochastic gradient-based methods.

In the (hypothetical) second part of this series, we'll draw an interesting connection between data compression and variational inference.

## References

<a id="1">[1]</a> 
Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur.,
"High-Fidelity Generative Image Compression.",
arXiv:2006.09965 (2020).

<a id="2">[2]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston.,
"Variational image compression with a scale hyperprior",
arXiv:1802.01436 (2018).

![grass](/assets/images/grass_again.jpg)
