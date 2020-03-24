# Problem in DA for Unobserved Link Volume Estimation (ULVE)

* What causes the domain difference, or joint distribution shift ?

  1. the random choice of individual vehicles

  2. volume observer would be incline to installed in critical intersections (normally with larger demand)

     // is this sample selection bias? or could we say the posterior prob is the same for different links ?

     > Since the underlying data-generating distribution remains constant, the underlying posteriors and conditionals remain equivalent between the biased and unbiased samples

     // is the data-generating distribution constant for different links ?

  3. the internal variance of links (length, lane width, etc.) in road networks



* sample-based method
  * weight the source samples
    * how to weight? 
      * kernel density approximation, and get the probability
      * train a classifier with logits, and get the probability.