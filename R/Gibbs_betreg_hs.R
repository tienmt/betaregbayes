#' Gibbs Sampler for Beta Regression with Horseshoe Prior
#'
#' Implements a Gibbs sampler for Bayesian beta regression using a Horseshoe prior.
#'
#' @param y Numeric vector. Response variable (must be in (0,1), transformed if needed).
#' @param X Numeric matrix. Design matrix of predictors.
#' @param phi Numeric. Precision parameter of the Beta likelihood (often fixed).
#' @param n_iter Integer. Number of MCMC iterations.
#' @param burn_in Integer. Number of burn-in iterations to discard from posterior summaries.
#' @param level Numeric. Credible interval level (default is 0.95).
#'
#' @details
#' This function performs Bayesian inference for a beta regression model with a Horseshoe prior on the regression coefficients.
#'
#' **Beta Regression** is suitable for modeling continuous response variables that lie strictly between 0 and 1, such as proportions or rates.
#' The model assumes that the response variable \eqn{y_i \in (0, 1)} follows a Beta distribution:
#' \deqn{y_i \sim \text{Beta}(\mu_i \phi, (1 - \mu_i)\phi)}
#' where \eqn{\mu_i = \text{logit}^{-1}(x_i^\top \beta)} is the mean of the Beta distribution, modeled via a logit link function,
#' and \eqn{\phi > 0} is a fixed or known precision parameter controlling the dispersion.
#'
#' **Horseshoe Prior** is a sparsity-inducing prior used for variable selection and shrinkage in high-dimensional settings.
#' Each regression coefficient \eqn{\beta_j} is assigned a hierarchical prior:
#' \deqn{
#'   \beta_j \sim \mathcal{N}(0, \lambda_j^2 \tau^2),
#'   \\
#'   \lambda_j \sim \text{Half-Cauchy}(0, 1),
#'   \\
#'   \tau \sim \text{Half-Cauchy}(0, 1)
#' }
#' where \eqn{\lambda_j} are local shrinkage parameters and \eqn{\tau} is a global shrinkage parameter.
#' This setup enables strong shrinkage of noise coefficients while preserving signals.
#'
#' Polya-Gamma data augmentation (via the `BayesLogit` package) is used to sample from the conditional posterior of \eqn{\beta}.
#'
#' @return A list with components:
#' \describe{
#'   \item{beta_samples}{Matrix of posterior samples for \eqn{\beta}.}
#'   \item{selected_variable}{Logical vector indicating whether the variable is "significant" (CI excludes 0).}
#'   \item{ci}{Matrix of credible intervals and medians for each coefficient.}
#' }
#' @references
#' Cribari-Neto, F., & Zeileis, A. (2010). Beta Regression in R. *Journal of Statistical Software*, 34(2), 1–24. https://doi.org/10.18637/jss.v034.i02
#'
#' Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2), 465–480. https://doi.org/10.1093/biomet/asq017
#'
#' Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian Inference for Logistic Models Using Pólya–Gamma Latent Variables. *Journal of the American Statistical Association*, 108(504), 1339–1349. https://doi.org/10.1080/01621459.2013.829001
#'
#' @author
#' The Tien Mai <the.tien.mai@fhi.no>
#'
#' @importFrom BayesLogit rpg
#' @importFrom mvtnorm rmvnorm
#' @importFrom stats rgamma quantile median
#' @examples
#'
#'   set.seed(42)
#'   n <- 100
#'   p <- 5
#'   X <- matrix(rnorm(n * p), n, p)
#'   beta_true <- c(2, -1.5, rep(0, p - 2))
#'   eta <- X %*% beta_true
#'   mu <- 1 / (1 + exp(-eta))
#'   phi <- 10
#'   y <- rbeta(n, mu * phi, (1 - mu) * phi)
#'
#'   res <- betareg_bayes(y, X, phi = 10, n_iter = 800, burn_in = 200)
#'   estimate_HS <- colMeans(res$beta_samples)
#'   mean( (estimate_HS - beta_true)^2 )
#'   res$selected_variable
#'   res$ci
#'
#' @export
betareg_bayes <- function(y,
                          X,
                          phi = 1,
                          n_iter = 1000,
                          burn_in = 0,
                          level = 0.95) {
  n <- length(y)
  p <- ncol(X)
  tX <- t(X)

  # Storage
  beta_samples <- matrix(0, nrow = n_iter, ncol = p)

  # Initialization
  beta <- rep(0, p)
  lambda2 <- rep(1, p)
  nu <- rep(1, p)
  tau2 <- 1
  xi <- 1

  for (iter in 1:n_iter) {
    # Step 1: Sample omega from Polya-Gamma
    eta <- X %*% beta
    omega <- BayesLogit::rpg(n = n, h = phi, z = eta)

    # Step 2: Sample beta | omega, HS prior
    kappa <- phi * (y - 0.5)
    Lambda_inv <- diag(1 / (lambda2 * tau2))
    V_beta <- solve( tX %*% (omega * X) + Lambda_inv)
    m_beta <- V_beta %*% (tX %*% kappa)
    beta <- as.vector(mvtnorm::rmvnorm(1, mean = m_beta, sigma = V_beta))

    # Step 3: Update local lambda2 and auxiliary nu
    for (j in 1:p) {
      lambda2[j] <- 1 / rgamma(1, shape = 1, rate = 1 / nu[j] + beta[j]^2 / (2 * tau2))
      nu[j] <- 1 / rgamma(1, shape = 1, rate = 1 + 1 / lambda2[j])
    }

    # Step 4: Update global tau2 and auxiliary xi
    tau2 <- 1 / rgamma(1, shape = (p + 1) / 2, rate = 1 / xi + sum(beta^2 / lambda2) / 2)
    xi <- 1 / rgamma(1, shape = 1, rate = 1 + 1 / tau2)

    # Store sample
    beta_samples[iter, ] <- beta
  }

  # Compute credible intervals
  posterior_samples <- beta_samples[(burn_in + 1):n_iter, , drop = FALSE]
  credible_intervals <- function(samples, level = 0.95) {
    alpha <- (1 - level) / 2
    ci_mat <- apply(samples, 2, function(x) {
      c(Lower = quantile(x, probs = alpha),
        Median = median(x),
        Upper = quantile(x, probs = 1 - alpha))
    })
    ci_mat <- t(ci_mat)
    colnames(ci_mat) <- c("Lower", "Median", "Upper")
    return(ci_mat)
  }

  ci <- credible_intervals(posterior_samples, level)
  selected <- !(ci[, "Lower"] < 0 & ci[, "Upper"] > 0)

  return(list(
    beta_samples = posterior_samples,
    selected_variable = selected,
    ci = ci
  ))
}
