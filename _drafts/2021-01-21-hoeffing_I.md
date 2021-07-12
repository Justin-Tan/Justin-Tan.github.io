---
layout: post
title: "Cardinal Sins I: Peeking at the Test Set"
date: 2021-01-25
type: note
categories: machine-learning, generalization
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: 
excerpt_separator: <!--more-->
---

Standard machine learning dogma holds that for a sufficiently large dataset, we divide it into a train/validation/test split. The train test is used for learning, the validation set is used for model selection, but the test set is your final estimate of model performance. The test set must be locked away until final model evaluation. You must not modify your model after witnessing performance on the test set. Here ends the sermon etc. <!--more-->

* Contents
{:toc}

Everything in the preamble is true - modifying your model based on the results of test set performance means your model is no longer independent of the test data and it no longer serves as an unbiased estimator of generalization error. However, there is a restricted sense in which you can perform model selection on the test set and maintain a valid estimate of generalization error. This is 



## References

<a id="1">[1]</a> 
David M. Blei, Alp Kucukelbir, Jon D. McAuliffe.,
"Variational Inference: A Review for Statisticians.",
arXiv:1601.00670 (2016).

<a id="2">[2]</a> 
James Townsend, Tom Bird, David Barber
"Practical Lossless Compression with Latent Variables using Bits Back Coding",
arXiv:1901.04866 (2019).

<a id="3">[3]</a> 
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston.,
"Variational image compression with a scale hyperprior",
arXiv:1802.01436 (2018).

![grass](/assets/images/shell_cushion.jpg)
