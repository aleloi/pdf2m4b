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
