---
layout: post
title: "Renormalization group flow for stringy nonlinear sigma models"
date: 2022-09-01
type: note
categories: physics, renormalization-group, string-theory
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: When is it possible to study the dynamics of two-dimensional nonlinear sigma models? The qualitative answer is seductively geometric, but getting quantitative is a different matter.
excerpt_separator: <!--more-->
---

* Contents
{:toc}

## Nonlinear sigma models

Nonlinear sigma models (NLSMs), named for an idea during the wild west of particle physics that didn't quite work out, appear in many places in high-energy/condensed matter physics. Here we will informally sketch the behaviour of stringy NLSMs under renormalization group (RG) flow. 

Heuristically, they outline the dynamics of a set of interacting fields living on some target manifold $(M,g)$, whose behaviour is parameterized by a lower-dimensional manifold $(\Sigma,h)$. In string theory the space $\Sigma$ is the worldsheet, a two-dimensional generalization of the worldline traced out by a point particle. The target space is $n$-dim spacetime with nontrivial curvature. The interpretation then is that a stringy NLSM describes how the string propagates through curved spacetime, subject to conditions imposed by the physics of the worldsheet $\Sigma$.

I suspect there is nothing here which is 'novel', or at least trivially deduced from existing literature. 

<div class='figure'>
    <img src="/assets/images/cymfd.png"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> A nonlinear sigma model describes the dynamics of a set of interacting fields which live on some manifold $\Sigma$ (green donut) and map to some target manifold $M$ (psychadelic bubble), with nontrivial curvature in general.
    </div>
</div>


<!-- One simple example of this is the $O(n)$ NLSM where the target space is the $(n-1)$-sphere $S^{n-1}$. This is used to describe spontaneous breaking of the the $O(n)$ rotational symmetry in the Heisenberg ferromagnet by the ground state, being spontaneously magnetized along some direction. -->

An infamous example is the generalization of the Polyakov action for the bosonic string to a curved spacetime. Let $X: \Sigma \rightarrow M$ be a map describing the coordinates of the string in spacetime. The classical action is the trace of pullback by $X$ of the metric on $M$ to $\Sigma$.

$$\begin{align}
    S &= \frac{1}{4 \pi \alpha'} \int_{\Sigma} \text{dVol} \, \textsf{Tr} \left[ X^* G \right]  \nonumber \\
    \label{eq:nlsm_action}
    &=  \frac{1}{4 \pi \alpha'} \int_{\Sigma} d^2 \sigma \sqrt{h} \, h^{\mu \nu} G_{ab}(X) \, \partial_{\mu} X^a \partial_{\nu} X^{b}.
\end{align}$$

- The scalar fields $X^a: \Sigma \rightarrow M, a = 1, \ldots n$ describe a surface in spacetime $M$ traced out by the string, essentially providing an embedding of $\Sigma$ in $M$.
- $h_{\mu \nu}$ is the metric on the 2D worldsheet $\Sigma$, coupled to the $X$s, and $G_{ab}(X)$ is the (nontrivial!) metric of the target manifold.
- $\sqrt{\alpha'}$ is the natural string length scale, recall historically the string tension is written $T = 1/(2\pi \alpha')$ and hence $\left[\alpha'\right] = -2$.

Here the target manifold metric $G_{ab}(X)$ plays the role of the coupling 'constants', which are field-dependent. This becomes important in RG flow later. The Polyakov action looks strange but agrees at the classical level with the geometric Nambu-Goto action, which is proportional to the area swept out by the worldsheet of the string. One of the more well-known curios of string theory is that this agreement survives quantization provided spacetime is 26 dimensional.

### Quantum fluctations

This section closely follows Chapter 7 of David Tong's notes [[1]](#1).

Geometrically, the NLSM action describes fluctations of fields living in the target manifold around some classical solution $x$, $X^a(\sigma) = x^a + \sqrt{\alpha'} Y^a(\sigma)$ - here the $\sqrt{\alpha'}$ factor is needed in order for $\left[ Y^a \right] = 0$, so the dynamical fluctuations $Y^a$ may be meaningfully taken to be small. Upon expanding the metric $G$ in $(\ref{eq:nlsm_action})$ we obtain:

$$\begin{equation}
    G_{ab}(X) \partial_{\mu} X^a \partial_{\nu} X^b = \alpha' \left( G_{\mu \nu}(x) + \sqrt{\alpha'} \partial_c G_{ab} Y^c + \frac{\alpha'}{2} \partial_c \partial_d G_{ab}Y^c Y^d + \cdots \right) \partial_{\mu}Y^a \partial_{\nu}Y^b.
\end{equation}$$

The coefficients of the fluctations $Y^a$ are the field-dependent coupling constants of the theory. If we want to employ perturbation theory, these must be small. When is this possible? Suppose the target manifold $M$ has a characteristic radius of curvature $r_c$. Schematically,

$$\begin{equation*}
    \frac{\partial G}{\partial X} \sim \frac{1}{r_c}.
\end{equation*}$$

Look at the leading order term. Morally, the dimensionless coupling that the expansion is conducted in is $\sqrt{\alpha'}/r_c$ - in order for perturbation theory to be valid, the spacetime metric must only vary on scales much greater than the characteristic length of the string. If $r_c \sim \sqrt{\alpha'}$, the higher-order terms become equally important as the lower-order terms, and we (literally) run into the strong coupling regime. The theory may not even be well-defined in this limit[^1]. At the very least it will need to be solved non-perturbatively. Nontrivial, to say the least.

[^1]: Explaining why theorists admonish us to respect the large-volume limit.


## Renormalization group flow

The spacetime metric $G_{ab}(X)$ may be interpreted as a functions' worth of coupling constants for the theory defined by the action $(\ref{eq:nlsm_action})$. This means that the metric depends on the energy scale $\mu$ (equivalently length scale $L = 1/\mu$) that we are probing our theory at, defined schematically via the beta function:

$$\begin{equation}
    \label{eq:betafn}
    \beta_{ab}(G) = \frac{\partial {G_{ab}(X; \mu)}}{ \partial \log \mu}.
\end{equation}$$

Note that here we are flowing from the IR to the UV with increasing $\mu$[^2]:

$$\begin{equation*}
    \textsf{IR} \xrightarrow{\hspace{5cm} \Large{\mu} \hspace{5cm}} \textsf{UV}
\end{equation*}$$

[^2]: Mathematicians who work on Ricci flow take the opposite convention, defining 'time' as $t = -\log(\mu/2\mu_0)$, then flowing from the UV to the IR in the positive direction of time, $\partial_t G_{ab} = - 2R_{ab}$. This seems faintly antisocial to me.

We now come to the intriguingly geometric crux of the matter: under the renormalization group, how the geometry of $M$ changes with the energy scale is dictated by the geometry itself. This was first demonstrated by Friedman in the 1980s [[2]](#2). The beta function for the theory $(\ref{eq:nlsm_action})$ is given perturbatively[^3] as:

[^3]: Cranking out the one-loop beta function is pretty standard, but things quickly get worse at higher orders. For the calculation of the stringy beta function up to four loops, see [[3]](#3).

$$\begin{equation}
    \beta_{ab} = \alpha' R_{ab} + \frac{\alpha'^2}{2} R_{a\mu\nu\rho}R_b^{\, \, \mu \nu \rho} + O(\alpha'^3).
\end{equation}$$

This is a very beautiful result - the evolution of the spacetime metric is inherently geometric. In order for the theory $(\ref{eq:nlsm_action})$ to be conformally invariant at the quantum level, we can't have a preferred length scale in the game. i.e. physical observables cannot depend on the scale $\mu$. If we want our theory to be a conformal field theory, we require that the spacetime $M$ satisfies (at least at one loop): 

$$\begin{equation}
    \beta_{ab} = R_{ab} = 0.
\end{equation}$$

The infamous Ricci-flat condition! This tells us that to leading order, the background spacetime through which the string propagates must obey the Einstein equations in vacuum.


### All the way with Chern-Gauss-Bonnet

Previously, we saw that the evolution of the metric $G_{ab}(X)$ is determined locally by the curvature. The moral of this section is that the evolution of the volume, a global quantity, is determined by a topological invariant, the Euler characteristic. The bridge here is the Chern-Gauss-Bonnet theorem, which converts local geometric information into something global.

We usually make the assumption that $M$ is compact and has finite volume

$$\begin{equation*}
    V = \int_M d^n X \, \sqrt{G(X)}.
\end{equation*}$$

The metric evolves with scale, so naturally the volume measured by the metric does as well. Using the identity $\partial_{\lambda} \sqrt{G} = \frac{1}{2} \sqrt{G} G^{ab}\partial_{\lambda} G_{ab}$, we observe how the volume evolves under RG flow:

$$\begin{align*}
    \frac{\partial V}{\partial \log \mu} &= \int_M d^n X \, \frac{\partial \sqrt{G(X)}}{\partial \log \mu} \\
    &= \frac{1}{2} \int_M d^n X \sqrt{G} \, G^{ab} \left( \frac{\alpha'}{2 \pi} R_{ab} + \frac{\alpha'^2}{8 \pi^2} R_{a \mu \nu \rho} R_b^{\, \,\mu \nu \rho} + O(\alpha'^3) \right) \\
    &= \frac{1}{4 \pi} \int_M d^n X \sqrt{G} \left( \alpha' R + \frac{\alpha'^2}{4\pi} \vert \textsf{Riem} \vert^2 + O(\alpha'^3) \right). 
\end{align*}$$

To one loop, the volume evolves under renormalization as the integral of the Ricci curvature over $M$, which is also coincidentally the Einstein-Hilbert action:

$$\begin{equation}
    \frac{\partial V}{\partial \log \mu} = \frac{\alpha'}{4 \pi} \int_M d^n X \sqrt{G}  \, R + O(\alpha'^2).
\end{equation}$$

* If $M$ is two dimensional, one may recognize the leading order term as the content of the classical Gauss-Bonnet theorem. This essentially relates geometry to topology, and says that the number of holes (genus $g$) of the surface is given by the integral of the curvature over the surface. 

    $$\begin{equation}
        \chi(M) = 2 - 2 g(M) = \frac{1}{2\pi} \int_M \textrm{dVol} \, R.
    \end{equation}$$
  
    So, to leading order, the evolution of the volume is dictated by a topological invariant, the number of holes in the surface!

  $$\begin{equation}
     \frac{\partial V}{\partial \log \mu} \approx \frac{\alpha'}{2} \chi(M) = \alpha'\left(1-g(M)\right)
    \end{equation}$$

    For instance, the $O(3)$ nonlinear sigma model has target space $S^2$, which has no holes - hence the volume expands under renormalization. Tori (donuts) have one hole and hence the volume remains fixed under RG flow. 

    **TL;DR**: The number of holes $g(M)$ in the Riemann surface controls whether spacetime expands/contracts as the energy scale runs up. How bizzare.

* In four dimensions the Chern-Gauss-Bonnet theorem states, modulo constant factors:

    $$\begin{equation}
        \chi(M) = \frac{1}{8 \pi^2} \int_M \textrm{dVol} \, \left(\vert \textsf{Riem} \vert^2 - 4 \vert \textsf{Ric} \vert + R^2 \right)
    \end{equation}$$

    This is not exactly the two-loop beta function, but for Ricci-flat manifolds $\textsf{Ric}=0 \implies R=0$. Hence the Euler characteristic dictates the evolution of the volume for two-loop RG flow on Ricci-flat four-dimensional manifolds:

    $$\begin{equation}
     \frac{\partial V}{\partial \log \mu} = \frac{\alpha'^2}{16 \pi^2} \int_M \textrm{dVol} \, \vert \textsf{Riem} \vert^2 + O(\alpha'^3) = -\frac{\alpha'^2}{2} \chi(M) + O(\alpha'^3)
    \end{equation}$$
    
    For non-Ricci-flat manifolds without a good deal of symmetry it is likely that the Kretschmann invariant $\textsf{Riem}^2$ dominates the integrand relative to the other factors. Is there an analogous topological invariant for the two-loop beta function for non Ricci-flat manifolds?

* In six dimensions (the number we like for Calabi-Yau compactification scenarios - at least for the heterotic superstring), the $O(\alpha'^3)$ term is in Eq. 2.27 of [[2]](#2) and looks fairly antisocial. I'm not sure whether there exists a nice explicit form for the $n=6$ Chern-Gauss-Bonnet theorem which cleanly reduces from the elementary definition of the Euler characteristic in terms of the Pfaffian of the curvature two-form $\chi(M) \propto \int_M \textsf{Pf}(\Omega)$.

## When can we perturb?

Recall that we are perturbing around a classical solution with effective coupling $\sqrt{\alpha'}/r_c$ - if $r_c \sim \sqrt{\alpha'}$ this breaks down - meaning if the volume shrinks under RG flow such that the radius of curvature approaches $\sqrt{\alpha'}$, our theory breaks down. When does this happen for general $M$? To one loop, we know 

$$\begin{equation}
    \frac{\partial V}{\partial \log \mu} \approx \frac{\alpha'}{4 \pi} \int_M d^n X \sqrt{G}  \, R.
\end{equation}$$

This will be the dominant contribution provided we are in the perturbative regime. But the evolution of the volume may kick us out after flowing along $\mu$. There are three cases to consider:

1. For $M$ with positive overall Ricci curvature, $M$ expands under the RG flow. Hence, at sufficiently high energies, the theory is asymptotically free and can be (probably) be solved perturbatively in the UV - depending on boundary conditions(?). At sufficiently low energies $r_c \sim \sqrt{\alpha'}$ and we are in the strong coupling regime. The theory **may** be QCD-like.

2. For $M$ with negative overall Ricci curvature, $M$ contracts under RG flow and becomes strongly coupled in the UV, but is probably perturbatively soluable in the IR - again depending on boundary conditions(?). At sufficiently high energies $r_c \sim \sqrt{\alpha'}$, where we cannot perturb. The theory **may** be QED-like. 

3. For a Ricci-flat manifold with overall zero Ricci curvature, the volume does not evolve under RG flow to leading order, and we are at a critical point. The theory is conformally invariant at one-loop level in this scenario. This is just what we want - more on this at a later date...

I'm not sure how we can quantitatively predict the breakdown scale about any of these cases without detailed knowledge of the theory. I'm not even sure what more can be said qualitatively about breakdown of these models at strong coupling. "_All happy families are alike; each unhappy family is unhappy in its own way_" - and non-perturbative theories are the unhappiest families in all of physics.

<h2> $\mathfrak{? ? ?}$ </h2>

* At what scale does perturbation theory break down in the strong coupling limit? Can this be made quantitative for simple models? Is this related to formation of singularities during the Ricci-flow?
* Sigma models with Ricci-flat target manifold are conformally invariant to leading order. Higher-loop corrections appear to break conformal symmetry - what does this mean for consistency of the theory?
* How does bringing in higher-loop corrections change the qualitative behaviour of the beta function? Does RG flow at higher loops admit a corresponding geometric interpretation?
* Does the renormalization of the volume make sense in the Kähler setting where the volume is fixed by choice of the Kähler moduli?

<h2> References </h2>

<a id="1">[1]</a> 
D. Tong. "[Lectures on String Theory](https://www.damtp.cam.ac.uk/user/tong/string.html)", Cambridge, 2009.

<a id="2">[2]</a> 
D. Friedan. "[Nonlinear Models in $2+\epsilon$ dimensions](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.45.1057)", Phys. Rev. Lett. 45, 1057, September 1980.

<a id="3">[3]</a> 
I. Jack, D.R.T. Jones, N. Mohammedi. "[A Four Loop Calculation of the Metric Beta Function for the Bosonic σ Model and the String Effective Action](https://inspirehep.net/literature/25100)", Nucl. Phys. B 322, 431-470, February 1989.

![grass](/assets/images/shell_zzZZz.jpg)
