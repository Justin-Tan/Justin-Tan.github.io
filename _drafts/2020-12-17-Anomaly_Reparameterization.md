---
layout: post
title: "Anomaly Detection and Invertible Reparameterizations"
date: 2020-12-17
categories: machine-learning, anomaly-detection
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: Density-based anomaly detection methods identify anomalies in the data as points with low probability density under the learned model. However, an invertible reparameterization of the data can arbitrarily change the density levels of any point - should anomalous points in one representation retain this status in any other representation?
excerpt_separator: <!--more-->
---

A recent [paper](https://openreview.net/pdf?id=6rTxPcEL7-) (Le Lan, Dinh, 2020) [[1]](#1) at the nicely themed ["I Can't Believe it's not Better" workshop](https://i-cant-believe-its-not-better.github.io/) at NeurIPS2020 asserts the following principle: "_In an infinite data and capacity setting, the result of an anomaly detection method should be invariant to any continuous invertible reparametrization_ ..." i.e. the status of an "anomaly" should be preserved across any choice of invertible reparameterization. We investigate arguments for and against this principle here.<!--more-->

* Contents
{:toc}

## 1. Preliminaries
One can change the data representation via some invertible map $f: \mathcal{X} \rightarrow \mathcal{X}$ and investigate the associated change in the probability density via the change of variables. 


## References

<a id="1">[1]</a> 
Le Lan, Charline and Dinh, Laurent.
"[Perfect density models cannot guarantee anomaly detection.](https://openreview.net/pdf?id=6rTxPcEL7-)",
I Can't Believe it's not Better Workshop, NeurIPS 2020.

<!-- ![grass](/assets/images/grass_again.jpg) -->
