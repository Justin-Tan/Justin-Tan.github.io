---
layout: post
title: "A non-constructive proof of the fundamental theorem of Riemannian geometry"
date: 2022-12-15
type: note
categories: differential-geometry
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: An interesting non-constructive proof of the fundamental theorem of Riemannian geometry, written to put off doing actual work. 
excerpt_separator: <!--more-->
---

<!-- * Contents
{:toc} -->

In a torpor of snow-induced boredom, here is an interesting non-constructive proof of the fundamental theorem of Riemannian geometry I learned from Jack Smith at Cambridge. The standard constructive proof demonstrates the Levi-Civita connection is really the only possibility for an orthogonal, torsion-free connection on a Riemannian manifold, but here's a more interesting but less useful proof.

## Statement

Recall a connection $\mathscr{A}$ on $(E,g)$ is _orthogonal_ if the inner product is covariantly constant, $d^{\mathscr{A}} g = 0$. Consider an $n$-dim Riemannian manifold $X$ and take the vector bundle $E$ to be the tangent bundle $TX$. We then have the: 

**$\color{#FAEBD7}{\textsf{Theorem (Fundamental theorem of Riemannian geometry)}}$**
: * Fix a Riemannian manifold $(X,g)$. There exists a unique, torsion-free orthogonal connection on $TX$. 

This is nice because once we have the Riemannian metric $g$, we have a distinguished connection. A connection on $TX$ induces connections in all tensor bundles $T^{(r,s)} X$ over $X$. Once equipped with $g$ we may differentiate any tensor bundle section to our heart's content in a 'canonical' manner.

## Proof

The idea is to show the more general statement that the map 

$$\begin{equation}
  \label{eq:og_map}
  \left\{\textsf{Orthogonal connections}\right\} \rightarrow \Omega^2(TX) = \Gamma(TX) \otimes \Omega^2(X),
\end{equation}$$

sending a connection to its torsion is a bijection. In other words, for any $TX$-valued two-form, $\exists$ a unique orthogonal connection which has this as its torsion. The theorem follows by taking the torsion to be zero.

Fix an arbitrary orthogonal connection $\mathscr{A}_0$. The space of orthogonal connections is an affine space for $\Omega^1(\mathfrak{o}(E))$, where $\mathfrak{o}(E) \leq \textsf{End}(E)$ denotes the bundle of skew-adjoint endomorphisms of $E$. Hence if $\Delta$ is an $\mathfrak{o}(E)$-valued one-form, $\Delta \in \Omega\left(\mathfrak{o}(E)\right)$, then any other orthogonal connection $\mathscr{A}$ may be written uniquely as $\mathscr{A} = \mathscr{A}_0 + \Delta$.

It is easier to prove that the map sending $\Delta$ to the difference of the associated torsions,

$$\begin{align}
  \label{eq:diff_map}
  f: \Omega^1(\mathfrak{o}(E)) &\rightarrow \Omega^2(E) \\
     \Delta &\mapsto T_{\mathscr{A}_0 + \Delta} - T_{\mathscr{A}_0} \nonumber
\end{align}$$

is a bijection. The map $f$ is linear, as opposed to the map in Eq. $\ref{eq:og_map}$, as the set of connections is not a vector space. Recall in some local trivialization the connection one-form is given by $\Gamma^i_{jk}dx^k$, where $(i,j)$ are the endomorphism indices. Explicitly, if the coordinates of $\mathscr{A}\_0$ are locally $\Gamma^i_{jk}$,

$$\begin{align*}
  \left( T_{\mathscr{A}_0 + \Delta} - T_{\mathscr{A}_0} \right)^i_{jk} &= \left(\Gamma + \Delta\right)^i_{kj} - \left( \Gamma + \Delta\right)^i_{jk} - \left( \Gamma^i_{kj} - \Gamma^i_{jk} \right) \\
  &= \Delta^i_{kj} - \Delta^i_{jk}.
\end{align*}$$

Notice this map is given by wedging with the solder form[^1] $\theta$, $f: \Delta \mapsto \Delta \wedge \theta$. Recall that the solder form is a bundle-valued one-form, 

$$\begin{equation*} \theta \in \Omega^1(E) = \Gamma(E \otimes T^{*}X) = \Gamma\left(\textsf{Hom}\left(TX, E\right)\right).
\end{equation*}$$

If $E$ is taken to be the tangent bundle, then define the solder form as the fibrewise identity map given locally as $e_l \otimes dx^l$, where $e_l$ is the $l$-th standard basis vector in $\mathbb{R}^n$. As $\Delta$ is an $\textsf{End}(TX)$-valued one-form, 

[^1]: Recall further the solder form $\theta$ an $E$-valued one-form - a section of $E \otimes T^{*}X$, i.e. the vector bundle of linear maps $\theta: TX \rightarrow E$ such that $\theta_p$ is an isomorphism $\forall \, p \in M$. If $E$ is taken to be the tangent bundle $TX$, then this is defined as the fibrewise identity map on $TX$.

$$\begin{align*}
  \Delta \wedge \theta &= \Delta \wedge (e_j \otimes dx^j) \\
  &= \Delta^i_{jk} e_i \otimes dx^k \wedge dx^j = - \Delta^i_{jk} e_i \otimes dx^j \wedge dx^k.
\end{align*}$$

By antisymmetry $\left(\Delta \wedge \theta\right)^i_{jk} = \Delta^i_{kj} - \Delta^i_{jk}$. 

### Advanced index pushing

The map $f$ between sections of one bundle to sections of another bundle is induced by the bundle morphism given by wedging with $\theta$ fibrewise:

$$\begin{equation*}
  F: \mathfrak{o}(TX) \otimes T^{*}X \rightarrow TX \otimes \bigwedge\nolimits^2 T^{*}X.
\end{equation*}$$

To show the map in Eq. $\ref{eq:diff_map}$ is a bijection, it suffices to show $F$ is an isomorphism, which we can do fibrewise. The entire problem is reduced to linear algebra. First note both bundles have the same rank. Locally, we have:

$$\begin{align*}
  \mathfrak{o}(TX) \otimes T^{*}X &= \left\{ \Delta^i_{jk} \, \vert \, \Delta_{ijk} = -\Delta_{jik} \right\} \\
  TX \otimes \bigwedge\nolimits^2 T^{*}X &= \left\{ T^i_{jk} = -T^i_{kj} \right\}.
\end{align*}$$

So both have rank $n \cdot { n \choose 2}$. Now it's left to show that $\Delta \mapsto \Delta \wedge \theta$ is injective. i.e. that if $\Delta^i_{jk}$ satisfies $\Delta_{ijk} = -\Delta{jik}$ after lowering the $TX$ index with the metric, and lies in $\textsf{Ker}(F)$ fibrewise, then $\Delta = 0$. $\Delta$ being in $\textsf{Ker}(F)$ means it is symmetric in it's last two indices. Now the problem reduces to index-pushing: 

$$ \Delta_{ijk} = -\Delta_{jik} = -\Delta_{jki} = \Delta_{kji} = \Delta_{kij} = -\Delta_{ikj} = -\Delta_{ijk} = 0.$$

So considering maps from connections to torsions reduces to some calculation on fibres and index-pushing to prove the FTRG. As we all know, this is the Levi-Civita connection $\Gamma$ on $(X,g)$. Our proof doesn't give an explicit construction for $\Gamma$, but there's Wikipedia for that.

<h2>  </h2>


![grass](/assets/images/heron.jpg)
