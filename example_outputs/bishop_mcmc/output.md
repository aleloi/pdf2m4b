Now suppose we move from a maximum likelihood approach to a full Bayesian
treatment in which we wish to sample from the posterior distribution over the parameter vector θ. In principle, we would like to draw samples from the joint posterior
_p(θ, Z_ **X), but we shall suppose that this is computationally difficult. Suppose fur-**
_|_
ther that it is relatively straightforward to sample from the complete-data parameter
posterior p(θ **Z, X). This inspires the data augmentation algorithm, which alter-**
_|_
nates between two steps known as the I-step (imputation step, analogous to an E
step) and the P-step (posterior step, analogous to an M step).

IP Algorithm

**I-step. We wish to sample from p(Z** **X) but we cannot do this directly. We**
_|_
therefore note the relation

## �
_p(Z_ **X) =** _p(Z_ **_θ, X)p(θ_** **X) dθ** (11.30)
_|_ _|_ _|_

and hence for l = 1, . . ., L we first draw a sample θ[(][l][)] from the current estimate for p(θ **X), and then use this to draw a sample Z[(][l][)]** from p(Z **_θ[(][l][)], X)._**
_|_ _|_

**P-step. Given the relation**

## �
_p(θ_ **X) =** _p(θ_ **Z, X)p(Z** **X) dZ** (11.31)
_|_ _|_ _|_

we use the samples **Z[(][l][)]** obtained from the I-step to compute a revised
_{_ _}_
estimate of the posterior distribution over θ given by


_p(θ_ **X)**
_|_ _≃_ [1]

_L_


_L_
## �

_p(θ_ **Z[(][l][)], X).** (11.32)
_|_
_l=1_


By assumption, it will be feasible to sample from this approximation in the
I-step.

Note that we are making a (somewhat artificial) distinction between parameters θ
and hidden variables Z. From now on, we blur this distinction and focus simply on
the problem of drawing samples from a given posterior distribution.

# 11.2. Markov Chain Monte Carlo

In the previous section, we discussed the rejection sampling and importance sampling strategies for evaluating expectations of functions, and we saw that they suffer
from severe limitations particularly in spaces of high dimensionality. We therefore
turn in this section to a very general and powerful framework called Markov chain
Monte Carlo (MCMC), which allows sampling from a large class of distributions,


-----

and which scales well with the dimensionality of the sample space. Markov chain
Monte Carlo methods have their origins in physics (Metropolis and Ulam, 1949),
and it was only towards the end of the 1980s that they started to have a significant
impact in the field of statistics.
As with rejection and importance sampling, we again sample from a proposal
distribution. This time, however, we maintain a record of the current state z[(][τ] [)], and
the proposal distribution q(z **z[(][τ]** [)]) depends on this current state, and so the sequence
_|_
_Section 11.2.1_ of samples z[(1)], z[(2)], . . . forms a Markov chain. Again, if we write p(z) = �p(z)/Zp,

we will assume that _p(z) can readily be evaluated for any given value of z, although_
## �
the value of Zp may be unknown. The proposal distribution itself is chosen to be
sufficiently simple that it is straightforward to draw samples from it directly. At
each cycle of the algorithm, we generate a candidate sample z[⋆] from the proposal
distribution and then accept the sample according to an appropriate criterion.
In the basic Metropolis algorithm (Metropolis et al., 1953), we assume that the
proposal distribution is symmetric, that is q(zA|zB) = q(zB|zA) for all values of
**zA and zB. The candidate sample is then accepted with probability**


## �
_A(z[⋆], z[(][τ]_ [)]) = min 1, _p[�][p]([(]z[z][(][τ][⋆][)][)])_

## �


## �
_._ (11.33)


This can be achieved by choosing a random number u with uniform distribution over
the unit interval (0, 1) and then accepting the sample if A(z[⋆], z[(][τ] [)]) > u. Note that
if the step from z[(][τ] [)] to z[⋆] causes an increase in the value of p(z), then the candidate
point is certain to be kept.
If the candidate sample is accepted, then z[(][τ] [+1)] = z[⋆], otherwise the candidate
point z[⋆] is discarded, z[(][τ] [+1)] is set to z[(][τ] [)] and another candidate sample is drawn
from the distribution q(z **z[(][τ]** [+1)]). This is in contrast to rejection sampling, where re_|_
jected samples are simply discarded. In the Metropolis algorithm when a candidate
point is rejected, the previous sample is included instead in the final list of samples,
leading to multiple copies of samples. Of course, in a practical implementation,
only a single copy of each retained sample would be kept, along with an integer
weighting factor recording how many times that state appears. As we shall see, as
long as q(zA|zB) is positive for any values of zA and zB (this is a sufficient but
not necessary condition), the distribution of z[(][τ] [)] tends to p(z) as τ . It should
_→∞_
be emphasized, however, that the sequence z[(1)], z[(2)], . . . is not a set of independent
samples from p(z) because successive samples are highly correlated. If we wish to
obtain independent samples, then we can discard most of the sequence and just retain every M [th] sample. For M sufficiently large, the retained samples will for all
practical purposes be independent. Figure 11.9 shows a simple illustrative example of sampling from a two-dimensional Gaussian distribution using the Metropolis
algorithm in which the proposal distribution is an isotropic Gaussian.
Further insight into the nature of Markov chain Monte Carlo algorithms can be
gleaned by looking at the properties of a specific example, namely a simple random


-----

**Figure 11.9** A simple illustration using Metropo
3

lis algorithm to sample from a
Gaussian distribution whose one
standard-deviation contour is shown

2.5

by the ellipse. The proposal distribution is an isotropic Gaussian distribution whose standard deviation is 2
0.2. Steps that are accepted are
shown as green lines, and rejected
steps are shown in red. A total of 1.5
150 candidate samples are generated, of which 43 are rejected.

1

0.5

0
0 0.5 1 1.5 2 2.5 3

walk. Consider a state space z consisting of the integers, with probabilities

_p(z[(][τ]_ [+1)] = z[(][τ] [)]) = 0.5 (11.34)

_p(z[(][τ]_ [+1)] = z[(][τ] [)] + 1) = 0.25 (11.35)

_p(z[(][τ]_ [+1)] = z[(][τ] [)] 1) = 0.25 (11.36)
_−_

where z[(][τ] [)] denotes the state at step τ . If the initial state is z[(1)] = 0, then by symmetry the expected state at time τ will also be zero E[z[(][τ] [)]] = 0, and similarly it is
_Exercise 11.10_ easily seen that E[(z[(][τ] [)])[2]] = τ/2. Thus after τ steps, the random walk has only trav
elled a distance that on average is proportional to the square root of τ . This square
root dependence is typical of random walk behaviour and shows that random walks
are very inefficient in exploring the state space. As we shall see, a central goal in
designing Markov chain Monte Carlo methods is to avoid random walk behaviour.

## 11.2.1 Markov chains

Before discussing Markov chain Monte Carlo methods in more detail, it is useful to study some general properties of Markov chains in more detail. In particular,
we ask under what circumstances will a Markov chain converge to the desired distribution. A first-order Markov chain is defined to be a series of random variables
**z[(1)], . . ., z[(][M]** [)] such that the following conditional independence property holds for
_m_ 1, . . ., M 1
_∈{_ _−_ _}_

_p(z[(][m][+1)]_ **z[(1)], . . ., z[(][m][)]) = p(z[(][m][+1)]** **z[(][m][)]).** (11.37)
_|_ _|_

This of course can be represented as a directed graph in the form of a chain, an example of which is shown in Figure 8.38. We can then specify the Markov chain by
giving the probability distribution for the initial variable p(z[(0)]) together with the


-----

conditional probabilities for subsequent variables in the form of transition probabil_ities Tm(z[(][m][)], z[(][m][+1)]) ≡_ _p(z[(][m][+1)]|z[(][m][)]). A Markov chain is called homogeneous_
if the transition probabilities are the same for all m.
The marginal probability for a particular variable can be expressed in terms of
the marginal probability for the previous variable in the chain in the form

## �
_p(z[(][m][+1)]) =_ _p(z[(][m][+1)]_ **z[(][m][)])p(z[(][m][)]).** (11.38)

_|_
**z[(][m][)]**


A distribution is said to be invariant, or stationary, with respect to a Markov chain
if each step in the chain leaves that distribution invariant. Thus, for a homogeneous
Markov chain with transition probabilities T (z[′], z), the distribution p[⋆](z) is invariant
if
## �
_p[⋆](z) =_ _T_ (z[′], z)p[⋆](z[′]). (11.39)

**z[′]**

Note that a given Markov chain may have more than one invariant distribution. For
instance, if the transition probabilities are given by the identity transformation, then
any distribution will be invariant.
A sufficient (but not necessary) condition for ensuring that the required distribution p(z) is invariant is to choose the transition probabilities to satisfy the property
of detailed balance, defined by

_p[⋆](z)T_ (z, z[′]) = p[⋆](z[′])T (z[′], z) (11.40)

for the particular distribution p[⋆](z). It is easily seen that a transition probability
that satisfies detailed balance with respect to a particular distribution will leave that
distribution invariant, because

## � � �

_p[⋆](z[′])T_ (z[′], z) = _p[⋆](z)T_ (z, z[′]) = p[⋆](z) _p(z[′]_ **z) = p[⋆](z).** (11.41)

_|_

**z[′]** **z[′]** **z[′]**

A Markov chain that respects detailed balance is said to be reversible.
Our goal is to use Markov chains to sample from a given distribution. We can
achieve this if we set up a Markov chain such that the desired distribution is invariant.
However, we must also require that for m, the distribution p(z[(][m][)]) converges
_→∞_
to the required invariant distribution p[⋆](z), irrespective of the choice of initial distribution p(z[(0)]). This property is called ergodicity, and the invariant distribution
is then called the equilibrium distribution. Clearly, an ergodic Markov chain can
have only one equilibrium distribution. It can be shown that a homogeneous Markov
chain will be ergodic, subject only to weak restrictions on the invariant distribution
and the transition probabilities (Neal, 1993).
In practice we often construct the transition probabilities from a set of ‘base’
transitions B1, . . ., BK. This can be achieved through a mixture distribution of the
form

_K_
## �

_T_ (z[′], z) = _αkBk(z[′], z)_ (11.42)

_k_ 1


-----

for some set of mixing coefficients α1, . . ., αK satisfying αk ⩾ 0 and [�]k _[α][k][ = 1][.]_

Alternatively, the base transitions may be combined through successive application,
so that

## � �
_T_ (z[′], z) = _. . ._ _B1(z[′], z1) . . . BK−1(zK−2, zK−1)BK(zK−1, z). (11.43)_

**z1** **zn−1**


If a distribution is invariant with respect to each of the base transitions, then obviously it will also be invariant with respect to either of the T (z[′], z) given by (11.42)
or (11.43). For the case of the mixture (11.42), if each of the base transitions satisfies detailed balance, then the mixture transition T will also satisfy detailed balance. This does not hold for the transition probability constructed using (11.43), although by symmetrizing the order of application of the base transitions, in the form
_B1, B2, . . ., BK, BK, . . ., B2, B1, detailed balance can be restored. A common ex-_
ample of the use of composite transition probabilities is where each base transition
changes only a subset of the variables.

## 11.2.2 The Metropolis-Hastings algorithm

Earlier we introduced the basic Metropolis algorithm, without actually demonstrating that it samples from the required distribution. Before giving a proof, we
first discuss a generalization, known as the Metropolis-Hastings algorithm (Hastings, 1970), to the case where the proposal distribution is no longer a symmetric
function of its arguments. In particular at step τ of the algorithm, in which the current state is z[(][τ] [)], we draw a sample z[⋆] from the distribution qk(z|z[(][τ] [)]) and then
accept it with probability Ak(z[⋆], zτ ) where


## �
_Ak(z[⋆], z[(][τ]_ [)]) = min 1,

## �p[�][p]([(]z[z][(][τ][⋆][)][)])[q]q[k]k[(]([z]z[(][τ][⋆][)]|z[|][z][(][τ][⋆][)][)])


## �
_._ (11.44)


Here k labels the members of the set of possible transitions being considered. Again,
the evaluation of the acceptance criterion does not require knowledge of the normalizing constant Zp in the probability distribution p(z) = _p(z)/Zp. For a symmetric_
## �
proposal distribution the Metropolis-Hastings criterion (11.44) reduces to the standard Metropolis criterion given by (11.33).
We can show that p(z) is an invariant distribution of the Markov chain defined
by the Metropolis-Hastings algorithm by showing that detailed balance, defined by
(11.40), is satisfied. Using (11.44) we have

_p(z)qk(z|z[′])Ak(z[′], z)_ = min (p(z)qk(z|z[′]), p(z[′])qk(z[′]|z))
= min (p(z[′])qk(z[′]|z), p(z)qk(z|z[′]))
= _p(z[′])qk(z[′]|z)Ak(z, z[′])_ (11.45)

as required.
The specific choice of proposal distribution can have a marked effect on the
performance of the algorithm. For continuous state spaces, a common choice is a
Gaussian centred on the current state, leading to an important trade-off in determining the variance parameter of this distribution. If the variance is small, then the


-----

**Figure 11.10** Schematic illustration of the use of an isotropic
Gaussian proposal distribution (blue circle) to
sample from a correlated multivariate Gaussian
distribution (red ellipse) having very different standard deviations in different directions, using the
Metropolis-Hastings algorithm. In order to keep
the rejection rate low, the scale ρ of the proposal
distribution should be on the order of the smallest
standard deviation σmin, which leads to random
walk behaviour in which the number of steps separating states that are approximately independent
is of order (σmax/σmin)[2] where σmax is the largest
standard deviation.


_σmin_


proportion of accepted transitions will be high, but progress through the state space
takes the form of a slow random walk leading to long correlation times. However,
if the variance parameter is large, then the rejection rate will be high because, in the
kind of complex problems we are considering, many of the proposed steps will be
to states for which the probability p(z) is low. Consider a multivariate distribution
_p(z) having strong correlations between the components of z, as illustrated in Fig-_
ure 11.10. The scale ρ of the proposal distribution should be as large as possible
without incurring high rejection rates. This suggests that ρ should be of the same
order as the smallest length scale σmin. The system then explores the distribution
along the more extended direction by means of a random walk, and so the number
of steps to arrive at a state that is more or less independent of the original state is
of order (σmax/σmin)[2]. In fact in two dimensions, the increase in rejection rate as ρ
increases is offset by the larger steps sizes of those transitions that are accepted, and
more generally for a multivariate Gaussian the number of steps required to obtain
independent samples scales like (σmax/σ2)[2] where σ2 is the second-smallest standard deviation (Neal, 1993). These details aside, it remains the case that if the length
scales over which the distributions vary are very different in different directions, then
the Metropolis Hastings algorithm can have very slow convergence.

# 11.3. Gibbs Sampling

Gibbs sampling (Geman and Geman, 1984) is a simple and widely applicable Markov
chain Monte Carlo algorithm and can be seen as a special case of the MetropolisHastings algorithm.

Consider the distribution p(z) = p(z1, . . ., zM ) from which we wish to sample,
and suppose that we have chosen some initial state for the Markov chain. Each step
of the Gibbs sampling procedure involves replacing the value of one of the variables
by a value drawn from the distribution of that variable conditioned on the values of
the remaining variables. Thus we replace zi by a value drawn from the distribution
_p(zi|z\i), where zi denotes the i[th]_ component of z, and z\i denotes z1, . . ., zM but
with zi omitted. This procedure is repeated either by cycling through the variables


-----

**p** **g**

in some particular order or by choosing the variable to be updated at each step at
random from some distribution.
For example, suppose we have a distribution p(z1, z2, z3) over three variables,
and at step τ of the algorithm we have selected values z1[(][τ] [)][, z]2[(][τ] [)] and z3[(][τ] [)][. We first]
replace z1[(][τ] [)] by a new value z1[(][τ] [+1)] obtained by sampling from the conditional distribution
_p(z1|z2[(][τ]_ [)][, z]3[(][τ] [)][)][.] (11.46)

Next we replace z2[(][τ] [)] by a value z2[(][τ] [+1)] obtained by sampling from the conditional
distribution
_p(z2|z1[(][τ]_ [+1)], z3[(][τ] [)][)] (11.47)

so that the new value for z1 is used straight away in subsequent sampling steps. Then
we update z3 with a sample z3[(][τ] [+1)] drawn from

_p(z3|z1[(][τ]_ [+1)], z2[(][τ] [+1)]) (11.48)

and so on, cycling through the three variables in turn.

Gibbs Sampling

1. Initialize {zi : i = 1, . . ., M _}_

2. For τ = 1, . . ., T :

**– Sample z1[(][τ]** [+1)] _∼_ _p(z1|z2[(][τ]_ [)][, z]3[(][τ] [)][, . . ., z]M[(][τ] [)][)][.]

**– Sample z2[(][τ]** [+1)] _∼_ _p(z2|z1[(][τ]_ [+1)], z3[(][τ] [)][, . . ., z]M[(][τ] [)][)][.]
...
**– Sample zj[(][τ]** [+1)] _∼_ _p(zj|z1[(][τ]_ [+1)], . . ., zj[(][τ]−[+1)]1 _[, z]j[(][τ]+1[)]_ _[, . . ., z]M[(][τ]_ [)][)][.]
...
**– Sample zM[(][τ]** [+1)] _∼_ _p(zM_ _|z1[(][τ]_ [+1)], z2[(][τ] [+1)], . . ., zM[(][τ] [+1)]−1 [)][.]


# Josiah Willard Gibbs
1839–1903

Gibbs spent almost his entire life living in a house built by his father in
New Haven, Connecticut. In 1863,
Gibbs was granted the first PhD in
engineering in the United States,
and in 1871 he was appointed to
the first chair of mathematical physics in the United


States at Yale, a post for which he received no salary
because at the time he had no publications. He developed the field of vector analysis and made contributions to crystallography and planetary orbits. His
most famous work, entitled On the Equilibrium of Heterogeneous Substances, laid the foundations for the
science of physical chemistry.


-----

To show that this procedure samples from the required distribution, we first of
all note that the distribution p(z) is an invariant of each of the Gibbs sampling steps
individually and hence of the whole Markov chain. This follows from the fact that
when we sample from p(zi|{z\i), the marginal distribution p(z\i) is clearly invariant
because the value of z\i is unchanged. Also, each step by definition samples from the
correct conditional distribution p(zi|z\i). Because these conditional and marginal
distributions together specify the joint distribution, we see that the joint distribution
is itself invariant.
The second requirement to be satisfied in order that the Gibbs sampling procedure samples from the correct distribution is that it be ergodic. A sufficient condition
for ergodicity is that none of the conditional distributions be anywhere zero. If this
is the case, then any point in z space can be reached from any other point in a finite
number of steps involving one update of each of the component variables. If this
requirement is not satisfied, so that some of the conditional distributions have zeros,
then ergodicity, if it applies, must be proven explicitly.
The distribution of initial states must also be specified in order to complete the
algorithm, although samples drawn after many iterations will effectively become
independent of this distribution. Of course, successive samples from the Markov
chain will be highly correlated, and so to obtain samples that are nearly independent
it will be necessary to subsample the sequence.
We can obtain the Gibbs sampling procedure as a particular instance of the
Metropolis-Hastings algorithm as follows. Consider a Metropolis-Hastings sampling
step involving the variable zk in which the remaining variables z\k remain fixed, and
for which the transition probability from z to z[⋆] is given by qk(z[⋆]|z) = p(zk[⋆][|][z][\][k][)][.]
We note that z[⋆]\k [=][ z][\][k][ because these components are unchanged by the sampling]
step. Also, p(z) = p(zk|z\k)p(z\k). Thus the factor that determines the acceptance
probability in the Metropolis-Hastings (11.44) is given by

_A(z[⋆], z) =_ _[p][(][z][⋆][)][q][k][(][z][|][z][⋆][)]_ _p(zk[⋆][|][z][⋆]\k[)][p][(][z][⋆]\k[)][p][(][z][k][|][z][⋆]\k[)]_ (11.49)

_p(z)qk(z[⋆]|z) [=]_ _p(zk|z\k)p(z\k)p(zk[⋆][|][z][\][k][) = 1]_


where we have used z[⋆]\k [=][ z][\][k][. Thus the Metropolis-Hastings steps are always]
accepted.
As with the Metropolis algorithm, we can gain some insight into the behaviour of
Gibbs sampling by investigating its application to a Gaussian distribution. Consider
a correlated Gaussian in two variables, as illustrated in Figure 11.11, having conditional distributions of width l and marginal distributions of width L. The typical
step size is governed by the conditional distributions and will be of order l. Because
the state evolves according to a random walk, the number of steps needed to obtain
independent samples from the distribution will be of order (L/l)[2]. Of course if the
Gaussian distribution were uncorrelated, then the Gibbs sampling procedure would
be optimally efficient. For this simple problem, we could rotate the coordinate system in order to decorrelate the variables. However, in practical applications it will
generally be infeasible to find such transformations.
One approach to reducing random walk behaviour in Gibbs sampling is called
_over-relaxation (Adler, 1981). In its original form, this applies to problems for which_


-----

**p** **g**


**Figure 11.11** Illustration of Gibbs sampling by alternate updates of two variables whose
distribution is a correlated Gaussian.
The step size is governed by the standard deviation of the conditional distribution (green curve), and is O(l), leading to slow progress in the direction of
elongation of the joint distribution (red
ellipse). The number of steps needed
to obtain an independent sample from
the distribution is O((L/l)[2]).


_z2_


_z1_

the conditional distributions are Gaussian, which represents a more general class of
distributions than the multivariate Gaussian because, for example, the non-Gaussian
distribution p(z, y) exp( _z[2]y[2]) has Gaussian conditional distributions. At each_
_∝_ _−_
step of the Gibbs sampling algorithm, the conditional distribution for a particular
component zi has some mean µi and some variance σi[2][. In the over-relaxation frame-]

work, the value of zi is replaced with


_zi[′]_ [=][ µ][i] [+][ α][(][z][i] _[−]_ _[µ][i][) +][ σ][i][(1][ −]_ _[α]i[2][)][1][/][2][ν]_ (11.50)

where ν is a Gaussian random variable with zero mean and unit variance, and α
is a parameter such that 1 < α < 1. For α = 0, the method is equivalent to
_−_
standard Gibbs sampling, and for α < 0 the step is biased to the opposite side of the
mean. This step leaves the desired distribution invariant because if zi has mean µi
and variance σi[2][, then so too does][ z]i[′][. The effect of over-relaxation is to encourage]

directed motion through state space when the variables are highly correlated. The
framework of ordered over-relaxation (Neal, 1999) generalizes this approach to nonGaussian distributions.

The practical applicability of Gibbs sampling depends on the ease with which
samples can be drawn from the conditional distributions p(zk|z\k). In the case of
probability distributions specified using graphical models, the conditional distributions for individual nodes depend only on the variables in the corresponding Markov
blankets, as illustrated in Figure 11.12. For directed graphs, a wide choice of conditional distributions for the individual nodes conditioned on their parents will lead to
conditional distributions for Gibbs sampling that are log concave. The adaptive rejection sampling methods discussed in Section 11.1.3 therefore provide a framework
for Monte Carlo sampling from directed graphs with broad applicability.

If the graph is constructed using distributions from the exponential family, and
if the parent-child relationships preserve conjugacy, then the full conditional distributions arising in Gibbs sampling will have the same functional form as the orig

-----

**Figure 11.12** The Gibbs sampling method requires samples
to be drawn from the conditional distribution of a variable conditioned on the remaining variables. For graphical models, this
conditional distribution is a function only of the states of the
nodes in the Markov blanket. For an undirected graph this comprises the set of neighbours, as shown on the left, while for a
directed graph the Markov blanket comprises the parents, the
children, and the co-parents, as shown on the right.

inal conditional distributions (conditioned on the parents) defining each node, and
so standard sampling techniques can be employed. In general, the full conditional
distributions will be of a complex form that does not permit the use of standard sampling algorithms. However, if these conditionals are log concave, then sampling can
be done efficiently using adaptive rejection sampling (assuming the corresponding
variable is a scalar).
If, at each stage of the Gibbs sampling algorithm, instead of drawing a sample
from the corresponding conditional distribution, we make a point estimate of the
variable given by the maximum of the conditional distribution, then we obtain the
iterated conditional modes (ICM) algorithm discussed in Section 8.3.3. Thus ICM
can be seen as a greedy approximation to Gibbs sampling.
Because the basic Gibbs sampling technique considers one variable at a time,
there are strong dependencies between successive samples. At the opposite extreme,
if we could draw samples directly from the joint distribution (an operation that we
are supposing is intractable), then successive samples would be independent. We can
hope to improve on the simple Gibbs sampler by adopting an intermediate strategy in
which we sample successively from groups of variables rather than individual variables. This is achieved in the blocking Gibbs sampling algorithm by choosing blocks
of variables, not necessarily disjoint, and then sampling jointly from the variables in
each block in turn, conditioned on the remaining variables (Jensen et al., 1995).

# 11.4. Slice Sampling

We have seen that one of the difficulties with the Metropolis algorithm is the sensitivity to step size. If this is too small, the result is slow decorrelation due to random
walk behaviour, whereas if it is too large the result is inefficiency due to a high rejection rate. The technique of slice sampling (Neal, 2003) provides an adaptive step size
that is automatically adjusted to match the characteristics of the distribution. Again
it requires that we are able to evaluate the unnormalized distribution _p(z)._
## �
Consider first the univariate case. Slice sampling involves augmenting z with
an additional variable u and then drawing samples from the joint (z, u) space. We
shall see another example of this approach when we discuss hybrid Monte Carlo in
Section 11.5. The goal is to sample uniformly from the area under the distribution


-----

_p˜(z)_


_p˜(z)_


**p** **g**

_u_ _zmax_

_z_
_z[(][τ]_ [)]

(b)


_z_
_z[(][τ]_ [)]

(a)


**Figure 11.13** Illustration of slice sampling. (a) For a given value z[(][τ] [)], a value of u is chosen uniformly in
the region 0 ⩽ _u ⩽_ _p(z[(][τ]_ [)]), which then defines a ‘slice’ through the distribution, shown by the solid horizontal
e
lines. (b) Because it is infeasible to sample directly from a slice, a new sample of z is drawn from a region
_zmin ⩽_ _z ⩽_ _zmax, which contains the previous value z[(][τ]_ [)].

given by

_p(z, u) =_ �1/Zp if 0 ⩽ _u ⩽_ �p(z) (11.51)
## � 0 otherwise

where Zp = [�] �p(z) dz. The marginal distribution over z is given by


## �


_p(z, u) du =_
## �


du = [�][p]Z[(][z]p[)]


_p(z)_
## � e

0


1
_Zp_


= p(z) (11.52)


and so we can sample from p(z) by sampling from _p(z, u) and then ignoring the u_
## �
values. This can be achieved by alternately sampling z and u. Given the value of z
we evaluate _p(z) and then sample u uniformly in the range 0 ⩽_ _u ⩽_ _p(z), which is_
## � �
straightforward. Then we fix u and sample z uniformly from the ‘slice’ through the
distribution defined by _z :_ _p(z) > u_ . This is illustrated in Figure 11.13(a).
_{_ � _}_

In practice, it can be difficult to sample directly from a slice through the distribution and so instead we define a sampling scheme that leaves the uniform distribution
under _p(z, u) invariant, which can be achieved by ensuring that detailed balance is_
## �
satisfied. Suppose the current value of z is denoted z[(][τ] [)] and that we have obtained
a corresponding sample u. The next value of z is obtained by considering a region
_zmin ⩽_ _z ⩽_ _zmax that contains z[(][τ]_ [)]. It is in the choice of this region that the adaptation to the characteristic length scales of the distribution takes place. We want the
region to encompass as much of the slice as possible so as to allow large moves in z
space while having as little as possible of this region lying outside the slice, because
this makes the sampling less efficient.

One approach to the choice of region involves starting with a region containing
_z[(][τ]_ [)] having some width w and then testing each of the end points to see if they lie
within the slice. If either end point does not, then the region is extended in that
direction by increments of value w until the end point lies outside the region. A
candidate value z[′] is then chosen uniformly from this region, and if it lies within the
slice, then it forms z[(][τ] [+1)]. If it lies outside the slice, then the region is shrunk such
that z[′] forms an end point and such that the region still contains z[(][τ] [)]. Then another


-----

