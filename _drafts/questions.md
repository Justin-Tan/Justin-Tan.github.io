---
layout: post
title: "Questions"
date: 2020-12-27
categories: machine-learning, 
usemathjax: true
readtime: true
image: /assets/images/shell_web.jpg
subtitle: Some questions.
excerpt_separator: <!--more-->
---

Questions.

* Contents
{:toc}

# The Heilmeier Catechism

* What are you trying to do? Articulate your objectives using absolutely no jargon.
  * I'm interested in building generative models in the physical sciences. More specifically, I'm interested in building models which can approximate the probability density of configurations of systems in physics and chemistry.
  * In many problems in physics and chemistry, we know the exact underlying laws, but often it's computationally difficult or intractable to calculate interesting observables or simulate the evolution of systems through classical means. 
  * Using machine-learned densities is one way of finding an effective theory, or an approximate solution to many of these problems.
  * There is a lot of structure and symmetry in problems in physics and chemistry - for example, in many-body systems of interacting particles, the potential function, which governs the interactions between particles, should remain invariant under 3D rotations and translations.
  * We know that the system after a rotation and the original system are related by a very specific way, and are not arbitrarily different.
  * Models should take advantage of this prior knowledge instead of starting from a blank slate.
  * Parameter/data efficiency - system doesn't waste capacity on learning physical constraints if we can build these constraints in from the start.
  * Developing models which are aware of these prior constraints is an important step toward more trustworthy models that are guaranteed to return physically meaningful outputs.
  * Using models which are guaranteed to respect the symmetries of the problem is one important component of achieving physically meaningful outputs - it's a necessary but not sufficient condition!
  * This area is one step towards more physically trustworthy applications of ML in physics/chemistry.
  * In terms of how we can achieve this, I think that normalizing flows offer a very concrete avenue toward this objective.
  * Talk about NFs.
  * Problem reduces to designing equivariant transformations from a base density, which will be domain-dependent, but it's a relatively concrete and achievable route towards probability densities which are invariant under the symmetries of the problem.
* How is it done today, and what are the limits of current practice?
  * Much of the literature for density estimation for data from physics and chemistry do not use models which understand the symmetries of their respective problems.
  * Instead there's a trend to use architectures which are repurposed from applications to image data, which may be suboptimal.
  * If your model produces predictions which transform the wrong way under a true symmetry transform of your problem, then [problematic] - you'll have more trouble with guaranteeing your results are physically meaningful.
* What is new in your approach and why do you think it will be successful?
* Who cares? If you are successful, what difference will it make?
  * One big application of ML to physics and chemistry is to approximate very classically computationally intensive calculations.
  * One problem with this is how can we trust the results of these effective-theory models.
  * Using models which are guaranteed to respect the symmetries of the problem is one important component of achieving physically meaningful outputs - it's a necessary but not sufficient condition!
  * This area is one step towards more physically trustworthy applications of ML in the physical sciences.
* What are the risks?
* How much will it cost?
* How long will it take?
* What are the mid-term and final “exams” to check for success?
  * Testing proof of concept of models which preserve Lorentz symmetry on particle physics datasets consisting of energy-momentum vectors would be the most attainable goal I can think of.
  * Normalizing flow models offer a very concrete route toward achieving symmetry-aware density estimation, because the problem reduces to finding transformations which are equivariant to the symmetry.
  * I think there are very tangible starting directions and landmarks.

### What makes a great research statement?

A great statement is distinctive and shows awareness, curiosity, and forward thinking.

*Distinctiveness:*  Lots of statements talk about the “hottest” trends in the field right now. Successful Ph.D. students engage with trends but don’t follow them. Extremely successful Ph.D. students start new trends.

*Awareness:*  A strong statement illustrates that an applicant has read widely about work in the field (and/or related fields) and has a sense of the landscape. A very strong statement shows critical thinking, maybe a bit of skepticism, and a sense that the person is developing taste in problems.

*Curiosity:*  A strong statement shows that the person is motivated to seek out new knowledge. Clear research questions, even bold ones, are great; even better is a clear understanding of what concrete steps will need to be taken to answer those questions. The Heilmeier Catechism is a tool I often use when planning projects.

*Forward thinking:*  An applicant’s CV and letters help us see what they’ve already done. The research statement is an opportunity to get a sense of what kind of work they want to do over the next few years.  Most people won’t follow through exactly on the plan they describe here, but it’s important to have a plan nonetheless.

# High-Energy Theory (Challenger Mishra)

**Idea:** Find compactification of string theory which yields an effective theory close to the SM - many possible compactifications, exhaustive classification impossible. Use ML to:
* Explore systematically a space of possibilities to find an approximate pattern from which mathematical formulae can be deduced.

## Mathematical Phenomenology
* Can help to discover mathematical structures within string theory?
    * Note patterns then look for an explanation.
    * Black box ML gives PAC answers - filter interesting geometries to investigate further.
    * Want to interpret workings of black box predictions.

* String theory leads to very large data sets - many solutions to string theory.
* Can ML help with the large amount of string data - learn string theory SMs - distinguish physically relevant string theory models.
* Data is mathematical, given in terms of integer-valued tensors.

## QFT:
* A field has a value at every point in spacetime, particles are local excitations of these fields. To define a QFT, specify the fields and how they interact.

## Data Explosion

* Large database (~500,000,000) of Calabi-Yau manifolds - mathematical data which describes the "string landscape".
* Typical problem in string theory has form: `Integer Tensor Input` &rarr; `Integer Output`.
* Computational geometry has developed (expensive) algorithms to obtain the output from given input.
* Estimate the Hodge numbers of CICYs and WP4 hypersurfaces using supervised data from algebraic geometry.
    * Try to predict genus of topology of string geometries - there exist topological invariants which count the number of holes in each geometry.
    * ML makes predictions polynomial rather than doubly exponential in time when using tools from algebraic geometry.

## Predicting Baryons from Mesons
* Witten: Mesons are almost free, Baryons are excitations - learn mesonic data and predict baryonic data.
* 196 mesons, 43 baryons - machine learn baryon spectrum from mesons.
* How does a black box learn semantics without knowing syntax?
* Can we find analytic expressions by opening the black box?

## Questions
* How much do you trust these predictions - if your model predicts like a pentaquark mass at a certain value, would you go and tell CERN to search around that energy?
* Is there any work being done on interpreting the predictions of these models? I guess you're trying to find something like an effective theory that approximates the true behaviour.
* Is it important to understand the predictions of these machine learning programs in your field, or are you happy with black-box predictions?

# Materials Science (Bingqing Cheng)

**Idea:** Data-driven estimation of the chemical properties of materials - how does this extrapolate to conditions outside the training set? How can we verify the machine-learned predictions?

## Ab initio thermodynamics of liquid and solid water

* Use water as a model system for computational methods.
* _Ab-initio methods_: Predict properties of material starting from laws of quantum mechanics.
* Predicting thermodynamic properties of water ab-initio is non-trivial - high computational cost.
* Use ML to model atomic interactions to avoid expensive calculations. Marry classical methods with data-driven machine-learned interatomic potentials.
* Describe total energy of system as the sum of individual contributions from atom-centered environments.
* Energy and forces associated with a central atom determined by neighbours, long-range interactions are approximated in a mean-field manner.

## Evidence for supercritical behaviour of high-pressure liquid hydrogen

* Uses machine learning to model interatomic interactions of dense hydrogen to make predictions at low computational cost - i.e. machine-learn the potential fields.
* Simulations using machine-learned potentials predict a continuous molecular-to-atomic transition in the liquid.

## Questions
1. How can one trust the predictions of these machine learning models under conditions very different from the training data - e.g. at high temperatures or pressures for which there is no experimental data? in other words, how can one guarantee that the predictions returned by these methods are physically realistic? (If I understand correctly, these predictions still have to be experimentally validated, and this is difficult for conditions of very high temperature and pressure).
2. What is the main benefit of making these predictions - does machine learning lead to any new physical insight about materials that we can't get from classical simulators?
3. As a follow-up to (1) - how would you experimentally verify the predictions of your machine learning model in conditions that cannot be recreated experimentally or which cannot be simulated?
4. In your recent work, [[1]](#1) used machine learning to model the phase transition of hydrogen under high pressure. As I understand, please correct me if I'm wrong, these methods have to extrapolate from pre-existing training data, so how can you be confident that the results are realistic/physical? In another of your papers, [[2]](#2), you were able to predict the chemical properties of water from first principles, but these properties are much easier to verify than the properties of metallic hydrogen.

# Notes
* In the physical sciences, scientific knowledge is often expressed in the form of physically motivated likelihood functions and priors.
* Resulting density surfaces can be very complicated, multi-modal, discontinuous, have mass concentrated in small parameter regions, and result in prohibitively slow inference.
* Representative samples can be difficult to obtain due to small typical sets, requiring a prohibitively large number of function evaluations.

# Q&A

## Research Interests

### Previous Research

* My background is in experimental high-energy physics, looking at ways to improve the sensitivity of particle accelerator experiments to hypothetical signatures of new physics. My Master's degree was concerned with developing methods to detect a very rare class of decays at particle colliders, the technical term for these are penguin decays. During my analysis I became more interested in a more general problem which is common to many experimental analyses in particle physics. This problem was that naïvely applying supervised machine learning methods in these analyses can introduce a large bias into the distribution of physically important observables, such as the particle mass, and this can have a negative effect on the final uncertainty of the measurements we'd like to make.

* I was working on finding representations of the original data such that you can do all the regular tasks you would do in an analysis without introducing bias into the spectrum of physically important observables. My work framed this in terms of identifying an representation that is approximately invariant with respect to certain physical observables related to the original data. We showed that you can use the concept of disentangled representations in variational autoencoders, where the dimensions of the latent representation are enforced to be approximately independent - you can use this concept to isolate the influence of the observables into a subspace of the learned latent representation of the generative model. From this, you can obtain a representation which is safe to use by removing the corresponding subspace in this latent representation.

* This allows downstream tasks to safely access a relevant 'subset' of the observed information without being influenced by the physical observables we would like to preserve, eliminating a potential source of significant systematic uncertainty in analyses. This is closely related to the problem of algorithmic fairness, in both cases you would like predictions to be agnostic to certain characteristics of your input data.

* This work was originally part of a PhD project concerned with precision measurements of rare decays at the Belle II experiment in Tsukaba, Japan. About one year into the program, I realized I was more motivated by the challenging machine learning aspect of the project and transitioned to a Master's by research degree, with the objective of undertaking graduate study more oriented around machine learning.

### Current Research Interests

* More recently, I'm become interested in the application of generative models to the physical sciences. More specifically, I'm interested in generative models which can perform accurate density estimation on scientific data. I was originally interested in this in the context of identifying rare events in particle physics as points of low density assigned by generative models - this can be applied to the detection of hypothetical signatures of new physics in particle physics, which we think should appear as anomalous events amidst a large background of normal events. But there are many applications of good density estimators to a lot of areas in physics and chemistry beyond rare event detection - I think one area that's received a lot of attention is using machine learning to approximate generative processes that are expensive to simulate classically.

* So I worked in a couple projects trying to estimate densities well on particle physics data. The feature representations that the densities are estimated over are very traditional - designed in the 70s or 80s to distill as much low-level information as possible into something more informative. By rights, given enough model capacity, you should be able to do at least as well by using the energy and momentum of the observed particles in the event as your feature representation, as this in principle contains all the relevant information about the event. However, when I tried this, it didn't work as well as using higher-level features, which was surprising to me.

* One possible explanation I thought about was that all of these architectures for density estimation were developed for image data and originally used convolutional architectures to process the image inputs. These convolutional architectures were designed with translational symmetry of images in mind, so I'm curious if it's possible to design architectures for density estimation which is tailored to the structure of scientific data.

* We know that there's a lot of structure in scientific data - in many problems in physics and chemistry, we know the system which generates the data obeys some sort of physical symmetry, under which the problem does not fundamentally change. For example, for a system of interacting particles, such as in molecular dynamics, the potential energy of the system doesn't change if you rotate or translate your frame of reference, and a good model should recognize there isn't an arbitrary difference between a system after a 3D rotation and the original system, but that they are related in a specific manner. You can also have more abstract symmetries such as Gauge symmetry in high-energy theory and condensed matter, and Lorentz symmetry in particle physics.

* I think that there's a lot to gain, and I think it's been shown in multiple fields, that directly building in these symmetries into machine learning models leads to improved modelling of data in that specific data which hopefully can be used to improve applications of density estimation to the physical sciences.

* I think this is a very good example of how infusing machine learning models with domain-knowledge leads to a lot of very interesting applications.

### How can you be confident you will finish out your PhD this time?

* That is a very good question. I think the original decision to split was a consequence of diverging interests.
* I was almost entirely self-taught in machine learning, the more I read about it, the further my research topic was steered toward that direction.
* Eventually it reached the point where my supervisor suggested I change over from a PhD to an MPhil, because our research directions were very different, and I'm very grateful that he allowed me to do that.
* I don't exactly regret the time I spent there, because I learned a lot there, about communication and how to do research, and it oriented me towards this very interesting research direction.
* I know for a fact I'm very self-motivated to continue in this field and along this path, it's been about 3 years since I first started getting interested in machine learning.
* I'm also acutely aware that I need to finish what I start if I get a second chance at things - because I'm not getting any younger, if nothing else.

# References

<a id="1">[1]</a> 
Bingqing Cheng, Guglielmo Mazzola, Chris J. Pickard, Michele Ceriotti. (2020) Evidence for supercritical behaviour of high-pressure liquid hydrogen. Nature, 585, 217–220.

<a id="2">[2]</a> 
Bingqing Cheng, Edgar A Engel, Jörg Behler, Christoph Dellago, Michele Ceriotti. (2019) ab initio thermodynamics of liquid and solid water. Proceedings of the National Academy of Sciences, 116 (4), 1110-1115.
