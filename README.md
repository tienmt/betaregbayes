# betaregbayes

Bayesian sparse Beta regression for bounded response with Horseshoe prior via Gibbs sampler. 
This is based on the paper:  
**"Handling bounded response in high dimensions: a Horseshoe prior Bayesian Beta regression approach."**

## Installation

Install the package using:

```r
devtools::install_github('tienmt/betaregbayes')

library(betaregbayes)

# simulate data
   set.seed(42)
  n <- 100
   p <- 5
   X <- matrix(rnorm(n * p), n, p)
   beta_true <- c(2, -1.5, rep(0, p - 2))
   eta <- X %*% beta_true
   mu <- 1 / (1 + exp(-eta))
   phi <- 10
   y <- rbeta(n, mu * phi, (1 - mu) * phi)

  res <- betareg_bayes(y, X, phi = 10, n_iter = 800, burn_in = 200)
  ( estimate_HS <- colMeans(res$beta_samples) )
# find the ell_2 error to the true
   sum( (estimate_HS - beta_true)^2 )

# get the selected variable
   res$selected_variable

# get the credible intervals
   res$ci
