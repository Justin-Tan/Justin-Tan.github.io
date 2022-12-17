---
layout: post
title: "Blue-collared connections"
date: 2022-11-18
type: note
categories: physics, differential-geometry
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: blah blah connections blah blah
excerpt_separator: <!--more-->
---

<!-- * Contents
{:toc} -->

Vector bundles and the associated idea of a **connection** provide a natural language for dealing with internal symmetries in a field theory. Presumably it has other uses as well. The core idea is reasonably simple, the connection answers the question:

$$\begin{equation*}
\color{pink}{\textsf{How do we differentiate a section of a vector bundle?}}
\end{equation*}$$

Start with some vector bundle $E$ fibred over some base space $B$, which we denote $E \xrightarrow{\pi} B$. Consider a local section of $E$[^1], $s: B \rightarrow E$. Naïvely taking the derivative $ds$, we find to our dismay a dependence on our choice of local trivialization over an open set $U \subset B$. If you choose some trivialization and your friend chooses a different one, and you both start taking naïve derivatives, your efforts will end in disagreement. In mathspeak, there is no canonical way to take derivatives of sections of $E$.

[^1]: Thinking physically, for $E$ with fibres $F$ over the base $B$, sections of $E$ are just a set of scalar fields $\phi_i$ living in the spacetime $B$ taking values in the space $F$.

To take a derivative of a section in a civilized manner, one must introduce an additional structure on $E$ to ensure that the result of differentiation is 'consistent' between different choices of trivialization. Physically, this is just the statement that observables are independent of the choice of local trivialization. Physics would be strange if this were not true - Alice would have to choose her choice of local basis and communicate this to Bob, possibly light-years away, before they may conduct consistent experiments. Remarkably, from this simple caveat, the entire machinery of classical gauge theory falls out.

There is a straightforward way to define a connection which does not require the immediate introduction of strange differential operators or exotic subspaces of $TE$. This simply follows from compulsively requiring geometric objects to be well-defined, i.e. independent of choices made in the definition.

## Defining problems away

Fix a vector bundle of rank $k$, $E \xrightarrow{\pi} B$. A section $s$ may be viewed locally in $U_{\alpha} \subset B$ under a trivialization $\Phi_{\alpha}: \pi^{-1}\left(U_{\alpha}\right) \rightarrow U_{\alpha} \times \mathbb{R}^k$ as a $\mathbb{R}^k$-valued function $v_{\alpha}$.

The naïve derivative is $dv_{\alpha}$ - a vector-valued one-form/one-form-valued vector. This may be viewed as a local $E$-valued one-form via $\Phi_{\alpha}^{-1}$[^2]. How does this expression change if we compute the naïve derivative under a different trivialization $\Phi_{\beta}$ over some different $U_{\beta}$, and then transfer back to $\Phi_{\alpha}$? 

[^2]: More formally, a $E$-valued $r$-form on the vector bundle $E \rightarrow B$ is a section of $E \otimes \Lambda^r T^{\vee} B$. An $E$-valued $r$-form eats $r$ vectors in an antisymmetric multilinear fashion and spit out some element of the fibre $E_p$ over $p \in B$. Denote $\Omega^r(E) \triangleq \Gamma\left(E \otimes \Lambda^r T^{\vee} B)\right)$.

The derivative is the $\Phi_{\beta}$ trivialization is just $dv_{\beta} = d\left(g_{\beta \alpha}(b) v_{\alpha}\right)$, where the transition function $g_{\alpha\beta}: U_{\alpha} \cap U_{\beta} \rightarrow G$ takes values in the symmetry group of the fibre. Transferring back to the $\Phi_{\alpha}$ patch:

$$\begin{align*}
    \tilde{dv_{\alpha}} &= g_{\beta \alpha}^{-1} d(g_{\beta \alpha} v_{\alpha}) \\
    &= \color{red}{g_{\beta \alpha}^{-1} dg_{\beta \alpha} v_{\alpha}} + dv_{\alpha} \, \textsf{ (by Leibniz)}
\end{align*}$$

So $\tilde{dv_{\alpha}} \neq dv_{\alpha}$, with the term in red being the horrible coordinate-dependent error. We want a notion of differentiation that is agnostic to the choice of trivialization. This may be arranged by defining away the problem.


<div class="defn" text='Connection'>
  A connection $\mathscr{A}$ on $E$ 
</div>

## Physics-flavoured connections

The above discussion is along the lines of how physicists are exposed to connections. A typical discussion proceeds as follows - consider a set of fields $\phi_i, i = 1, \ldots n$ living on some spacetime $B$, transforming into one another under the action of some symmetry group $G$. Define a spacetime-dependent gauge transformation as:

$$ \tilde{\phi}_j(x) = g^i_j(x) \phi_i(x), $$

where $g: B \rightarrow G$ is an element of some representation of $G$.


<!-- The motivation is straightforward enough, although the presentation in most introductory texts leaves much to be desired. Grabbing a text at random from my shelf, the connection/covariant derivative is defined, as the following differential operator:

$$\begin{equation*}
    D: \Gamma(E) \rightarrow \Gamma(E) \otimes_{C^{\infty}(B)} \Gamma(T^{\vee}B),
\end{equation*}$$

satisfying $\mathbb{R}$-linearity and some Leibniz-type rule. Not particularly enlightening, or even atypical. In a haze of pedagogical zeal, some authors even introduce connections using vertical/horizontal subspaces of the tangent bundle of $E$. However, there is an embarassingly simple way to define a connection. Perhaps not the most conceptually satisfying, but comprehensible, at the least.  -->


## Motivation




<h2> References </h2>

<a id="1">[1]</a> 
D. Tong. "[Lectures on String Theory](https://www.damtp.cam.ac.uk/user/tong/string.html)", Cambridge, 2009.

<a id="2">[2]</a> 
D. Friedan. "[Nonlinear Models in $2+\epsilon$ dimensions](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.45.1057)", Phys. Rev. Lett. 45, 1057, September 1980.

<a id="3">[3]</a> 
I. Jack, D.R.T. Jones, N. Mohammedi. "[A Four Loop Calculation of the Metric Beta Function for the Bosonic σ Model and the String Effective Action](https://inspirehep.net/literature/25100)", Nucl. Phys. B 322, 431-470, February 1989.

<!-- ![grass](/assets/images/shell_zzZZz.jpg) -->
