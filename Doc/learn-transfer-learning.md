# Learn Transfer Learning

## 1 Domain Adaptation Review

> Kouw, W. M., & Loog, M. (2019). A review of domain adaptation without target labels. *arXiv e-prints*, arXiv:1901.05335

Transfer learning is all about generalization.

> Generalization is the process of observing a finite number of samples and making statements about all possible samples. 

### 1.0 Definitions and metrics

* Different domains: where the feature-label joint distribution is different, for single-source and single-target problem, the probability distribution $p_{S}(x,y)$ is different with $p_{T}(x,y)$
* Domain adaptation problem normally focus on the condition that distribution differs while transfer learning (not general transfer learning) focus more on commonalities.
* Two kinds of goals:
  1. transductive learning: predict the labels of given target features
  2. inductive learning: predict the labels of new target features

### 1.1 Sample-based methods

* focus on ==weighting individual observations== during training based on their importance to the target domain

  // how to measure the importance of a source sample $x_{src}^{(i)}$ to a target sample $x_{tar}^{(j)}$ ?

  // Two-step procedure: weight source samples $\rightarrow$ train a model with weighted samples

  * optimization-based: minimizing the distance between two feature distributions by decide the weights
    $$
    \hat{w}=\arg\min{D[w,p_{S}(x),p_{T}(x)]}
    $$
    The key is to select distance measure: MMD (Maximum Mean Distance), KL-divergence, etc.

  * direct weight estimator: knn, logistic regression, neural network, etc.

### 1.2 Feature-based methods

* revolve around on mapping, projecting and representing features such that a source classifier performs well on the target domain

  // Find a transformation (linear or non-linear) to generate effective and invariant features for both domains.



### 1.3 Inference-based methods

* incorporate adaptation into the parameter estimation procedure  