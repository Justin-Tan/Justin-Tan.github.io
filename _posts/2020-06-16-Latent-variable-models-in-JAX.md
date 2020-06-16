---
layout: post
title: "Latent Variable Models in Jax"
subtitle: "For density estimation/sampling."
date: 2020-06-16
categories: machine-learning, latent-variables, jax
usemathjax: true
readtime: True
image: /assets/images/shell_web.jpg
excerpt_separator: <!--more-->
---

In this post we'll examine applications of LVMs to density estimation/fast sampling. We'll implement two illustrative approaches in Jax, highlighting how Jax differs from standard Autodiff frameworks in some major aspects.<!--more--> In particular, we'll look at the recently proposed SUMO (Stochastically Unbiased Marginalization Objective). Finally, we end with the obligatory illustrative toy problem. There's a nice picture of my neighbour's cat at the end for some extra motivation. If you'd rather read code, check out the [associated repository on Github](https://github.com/justin-tan/density_estimation_jax). 

- SUMO
- Implementation in `Jax`
- Application to efficient sampling from "Neal's Funnel"
- Conclusion

* Contents
{:toc}

## 2. Density Estimation
{% highlight python %}
def diag_gaussian_entropy(mean, logvar):
    # Entropy of diagonal Gaussian
    D = mean.shape[0]
    diff_entropy = 0.5 * (D * jnp.log(2 * jnp.pi) + jnp.sum(logvar) + D)
    return diff_entropy
{% endhighlight %}

## 3. SUMO
In [an earlier post]({% post_url 2020-06-16-Intuitive-Importance-Weighted-ELBO-Bounds %}) we gave a heuristic motivation behind the importance-weighted autoencoder [[1]](#1) objective

### 3.1 Reducing Variance

Jax has some very interesting design choices. In particular, easy vectorization through `vmap`{:.python} helps to streamline your thought process a lot once you don't have to carry around a cumbersome batch dimension. Personally, there's not enough of a QoL improvement at the moment over Torch for me to port my existing libraries to Jax, but I do enjoy tinkering with it and its minimalism lends itself well to quick proof of concept sketches (provided you manage to avoid the sharp edges). Next we might implement some interesting normalizing flow variants, which could be really easy, or really troublesome, depending on if we can nail the behaviour of `jax.jit`{:.python}.

## References

<a id="1">[1]</a> 
Burda, Yuri and Grosse, Roger and Salakhutdinov, Ruslan.
Importance Weighted Autoencoders.
International Conference on Learning Representations (2016).