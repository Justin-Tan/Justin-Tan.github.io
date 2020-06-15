---
layout: post
title: "Latent Variable Models in Jax"
date: 2020-06-15
categories: machine-learning, latent-variables, jax
---

## Latent Variable Models in Jax
In this post we'll recap the basics of latent variable models for density estimation/sampling. Two approaches will be highlighted - the importance weighted autoencoder and the recently proposed SUMO (Stochastically Unbiased Marginalization Objective), an unbiased estimator of the log-marginal likelihood. Implementing these models in Jax will highlight how Jax differs from standard Autodiff frameworks in some major aspects. Finally, we end with the obligatory illustrative toy problem. There's a nice picture of my neighbour's cat at the end for some extra motivation. If you'd rather read code, check out the [associated repository on Github](https://github.com/justin-tan/density_estimation_jax). 

- Latent Variable Models
- The Importance Weighted ELBO
- SUMO
- Implementation in `Jax`
- Application to efficient sampling from "Neal's Funnel"
- Conclusion

## Latent Variable Models

Latent variable models allow us to augment observed data $x \in \mathcal{X}$ with unobserved, or latent variables $z \in \mathcal{Z}$. This is a modelling choice, in the hopes that we can better describe the observed data in terms of unobserved factors. We do this by defining a conditional model $p(x \vert z)$ that describes the dependence of observed data on latent variables, as well as a prior density $p(z)$ over the latent space. Typically defining this conditional model requires some degree of cleverness, or we can do the vogue thing and just use a big neural network.

Latent variables are useful because they may reveal something useful about the data. For instance, we may postulate that the high-dimensionality of the observed data may be 'artificial', in the sense that the observed data can be explained by more elementary, unobserved variables. Informally, we may hope that these latent factors capture most of the relevant information about the high-dimensional data in a succinct low-dimensional representation. To this end, we may make the modelling assumption that $$\text{dim}\left(\mathcal{Z}\right) \ll \text{dim}\left(\mathcal{X}\right)$$.

This is a test $\int_x p(y \vert x) p(x) $

Also a test \\[\int \pi /6 \\]

$$ f(x) = x^2 $$

$$\begin{multline}
\shoveleft
\begin{aligned}
G_t&=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 R_{t+4}+...\\
&=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4})+...\\
&=R_{t+1}+\gamma G_{t+1}\\
\end{aligned}
\end{multline}$$

{% highlight python %}
def diag_gaussian_entropy(mean, logvar):
    # Entropy of diagonal Gaussian
    D = mean.shape[0]
    diff_entropy = 0.5 * (D * jnp.log(2 * jnp.pi) + jnp.sum(logvar) + D)
    return diff_entropy
{% endhighlight %}


{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}


