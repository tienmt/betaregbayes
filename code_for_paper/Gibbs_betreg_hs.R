#
library(Rcpp); sourceCpp('test.cpp')

gibbs_sampler_betreg_HS <- function(y, X, phi, n_iter) {
  n <- length(y)
  p <- ncol(X)
  tX <- t(X)
  
  # Storage
  beta_samples <- matrix(0, nrow = n_iter, ncol = p)
  tau_sample <- c()
  
  # Initialization
  beta <- rep(0, p)
  lambda2 <- rep(1, p)
  nu <- rep(1, p)
  tau2 <- 1
  xi <- 1
  
  for (iter in 1:n_iter) {
    # Step 1: Sample omega from Polya-Gamma
    eta <- eigenMapMatMult(X , beta)
    omega <- BayesLogit::rpg(n = n, h = phi, z = eta)
    
    # Step 2: Sample beta | omega, HS prior
    kappa <- phi * (y - 0.5)
    Lambda_inv <- diag(1 / (lambda2 * tau2))
    V_beta <- eigen_inverse( eigenMapMatMult( tX, omega * X) + Lambda_inv + 0.001*diag(p) )
    m_beta <- eigenMapMatMult(V_beta , eigenMapMatMult(tX , kappa) )
    beta <- as.vector(mvtnorm::rmvnorm(1, mean = m_beta, sigma = V_beta))
    #beta <- m_beta + chol(V_beta) %*% rnorm(p)
    
    beta2222 <- beta^2
    # Step 3: Update local lambda2 and auxiliary nu
    lambda2 <- 1/ sapply(1:p, function(j) rgamma(1, shape = 1, rate = 1 / nu[j] + beta2222[j] / (2 * tau2)))
    nu <- 1/ sapply(1:p, function(j)rgamma(1, shape = 1, rate = 1 + 1 / lambda2[j]))
   # for (j in 1:p) {
  #    lambda2[j] <- 1 / rgamma(1, shape = 1, rate = 1 / nu[j] + beta2222[j] / (2 * tau2))
   #   nu[j] <- 1 / rgamma(1, shape = 1, rate = 1 + 1 / lambda2[j])
  #  }
    
    # Step 4: Update global tau2 and auxiliary xi
    tau2 <- 1 / rgamma(1, shape = (p + 1) / 2, rate = 1 / xi + sum(beta2222 / lambda2) / 2)
    xi <- 1 / rgamma(1, shape = 1, rate = 1 + 1 / tau2)
    # Store sample
    beta_samples[iter, ] <- beta
    tau_sample[iter] <- tau2
    
  }
  return(  return(list(beta_samples = beta_samples,tau = tau_sample))
)
}
gibbs_sampler_betreg_HS <- compiler::cmpfun(gibbs_sampler_betreg_HS)

selection_metrics <- function(true_support, selected_support, p = NULL) {
  # true_support: indices of truly relevant variables
  # selected_support: indices selected by your method
  # p: total number of variables (optional, required for TN)
  
  true_support <- as.integer(true_support)
  selected_support <- as.integer(selected_support)
  
  TP <- length(intersect(true_support, selected_support))
  FP <- length(setdiff(selected_support, true_support))
  FN <- length(setdiff(true_support, selected_support))
  if (!is.null(p)) {
    all_indices <- seq_len(p)
    TN <- length(setdiff(all_indices, union(true_support, selected_support)))
  } else {
    TN <- NA  # Cannot compute without knowing p
  }
  
  precision <- if ((TP + FP) == 0) NA else TP / (TP + FP)
  recall    <- if ((TP + FN) == 0) NA else TP / (TP + FN)
  specificity <- if (!is.na(TN) && (TN + FP) > 0) TN / (TN + FP) else NA
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    NA
  }
  fdr <- if ((TP + FP) == 0) 0 else FP / (TP + FP)
  
  return(c(
    'Precision' = precision,
    'Recall' = recall,
    'F1' = f1,
    'Specificity' = specificity,
    'FDR' = fdr
  ))
}
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

gibbs_sampler_betreg_HS_alpha <- function(y, X, phi, n_iter, alpha = 1) {
  n <- length(y)
  p <- ncol(X)
  tX <- t(X)
  
  # Storage
  beta_samples <- matrix(0, nrow = n_iter, ncol = p)
  tau_sample <- c()
  
  # Initialization
  beta <- rep(0, p)
  lambda2 <- rep(1, p)
  nu <- rep(1, p)
  tau2 <- 1
  xi <- 1
  
  for (iter in 1:n_iter) {
    # Step 1: Sample omega from tempered Polya-Gamma
    eta <- eigenMapMatMult(X, beta)
    omega <- BayesLogit::rpg(n = n, h = alpha * phi, z = eta)  # Tempered via alpha*phi
    
    # Step 2: Sample beta | omega, HS prior
    kappa <- alpha * phi * (y - 0.5)  # Scaled by alpha
    Lambda_inv <- diag(1 / (lambda2 * tau2))
    XtOmegaX <- eigenMapMatMult(tX, omega * X)
    V_beta <- eigen_inverse(XtOmegaX + Lambda_inv + 1e-6 * diag(p))  # Add jitter for stability
    m_beta <- eigenMapMatMult(V_beta, eigenMapMatMult(tX, kappa))
    beta <- as.vector(mvtnorm::rmvnorm(1, mean = m_beta, sigma = V_beta))
    
    # Step 3: Update local lambda2 and auxiliary nu
    beta_sq <- beta^2
    lambda2 <- 1 / sapply(1:p, function(j) rgamma(1, shape = 1, rate = 1 / nu[j] + beta_sq[j] / (2 * tau2)))
    nu <- 1 / sapply(1:p, function(j) rgamma(1, shape = 1, rate = 1 + 1 / lambda2[j]))
    
    # Step 4: Update global tau2 and auxiliary xi
    tau2 <- 1 / rgamma(1, shape = (p + 1) / 2, rate = 1 / xi + sum(beta_sq / lambda2) / 2)
    xi <- 1 / rgamma(1, shape = 1, rate = 1 + 1 / tau2)
    
    # Store sample
    beta_samples[iter, ] <- beta
  }
  return(list(beta_samples = beta_samples))
}
gibbs_sampler_betreg_HS_alpha <- compiler::cmpfun(gibbs_sampler_betreg_HS_alpha)
















betaregbayes_dir_laplace <- function(y,
                                     X,
                                     phi = 1,
                                     n_iter = 2000,
                                     burn_in = 0,
                                     level = 0.95,
                                     a = NULL) {
  # packages / checks
  if (!requireNamespace("BayesLogit", quietly = TRUE)) stop("Install BayesLogit")
  if (!requireNamespace("mvtnorm", quietly = TRUE)) stop("Install mvtnorm")
  if (!requireNamespace("statmod", quietly = TRUE)) stop("Install statmod")
  if (!requireNamespace("GIGrvg", quietly = TRUE)) stop("Install GIGrvg")
  
  n <- length(y)
  p <- ncol(X)
  tX <- t(X)
  # default concentration (common choice)
  if (is.null(a)) a <- 1 / p    
  
  # storage
  beta_samples <- matrix(0, nrow = n_iter, ncol = p)
  delta_samples <- matrix(0, nrow = n_iter, ncol = p) # delta_j = tau * phi_j
  psi_samples <- matrix(0, nrow = n_iter, ncol = p)
  
  # init
  beta <- rep(0, p)
  delta <- rep(1, p)
  psi <- rep(1, p)
  eps <- 1e-10
  
  for (iter in 1:n_iter) {
    ## 1) Polya-Gamma augmentation for Beta regression (same as before)
    eta <- X %*% beta
    omega <- BayesLogit::rpg(n = n, h = phi, z = as.numeric(eta))
    
    ## 2) Update beta | omega, psi, delta  (Gaussian)
    # prior variance for beta_j is psi_j * delta_j^2  => prior precision = 1/(psi * delta^2)
    prior_prec <- 1 / pmax(psi * (delta^2), eps)
    Q <- tX %*% (omega * X) + diag(as.numeric(prior_prec), p, p)
    kappa <- phi * (y - 0.5)
    # compute V_beta and m_beta robustly
    # try Cholesky; fall back to solve if needed
    chol_ok <- TRUE
    m_beta <- rep(0, p)
    V_beta <- tryCatch({
      R <- chol(Q)
      Vb <- chol2inv(R)
      m <- Vb %*% (tX %*% kappa)
      list(Vb = Vb, m = m)
    }, error = function(e) {
      chol_ok <<- FALSE
      Vb <- tryCatch(solve(Q), error = function(e2) {
        # jitter and try again
        solve(Q + diag(eps, p))
      })
      m <- Vb %*% (tX %*% kappa)
      list(Vb = Vb, m = m)
    })
    V_beta <- V_beta$Vb
    m_beta <- as.numeric(V_beta %*% (tX %*% kappa))
    beta <- as.numeric(mvtnorm::rmvnorm(1, mean = m_beta, sigma = V_beta))
    
    ## 3) Redundancy-free DL updates
    # 3a: delta_j | beta_j  ~ GIG(lambda = a - 1, chi = 2 * |beta_j|, psi = 1)
    #     (GIGrvg parametrisation: rgig(n, lambda, chi, psi) draws density ~ x^{lambda-1} exp(-(chi/x + psi * x)/2))
    for (j in 1:p) {
      chi_j <- 2 * max(abs(beta[j]), eps)   # chi >= 0
      # lambda can be < 0 when a < 1; GIGrvg supports that if chi > 0
      delta_j <- GIGrvg::rgig(1, lambda = a - 1, chi = chi_j, psi = 1)
      # numerical guard
      if (!is.finite(delta_j) || delta_j <= 0) delta_j <- 1e-6
      delta[j] <- delta_j
    
    # 3b: psi_e ~ Inv-Gaussian(mean = delta/|beta|, shape = 1); then psi = 1 / psi_e
    # (Onorati formulation: sample psi_e then invert)
      bj_abs <- max(abs(beta[j]), eps)
      mu_ig <- delta[j] / bj_abs
      ig_draw <- statmod::rinvgauss(1, mean = mu_ig, shape = 1)
      # invert (psi = 1 / psi_e)
      psi_j <- 1 / max(ig_draw, eps)
      if (!is.finite(psi_j) || psi_j <= 0) psi_j <- 1e-6
      psi[j] <- psi_j
    }
    
    ## store
    beta_samples[iter, ] <- beta
    delta_samples[iter, ] <- delta
    psi_samples[iter, ] <- psi
  }
  
  # posterior summaries after burn-in
  posterior_beta <- beta_samples[(burn_in + 1):n_iter, , drop = FALSE]
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
  ci <- credible_intervals(posterior_beta, level)
  selected <- !(ci[, "Lower"] < 0 & ci[, "Upper"] > 0)
  
  return(list(
    beta_samples = posterior_beta,
    delta_samples = delta_samples[(burn_in + 1):n_iter, , drop = FALSE],
    psi_samples = psi_samples[(burn_in + 1):n_iter, , drop = FALSE],
    selected_variable = selected,
    ci = ci
  ))
}











betaregbayes_SnS <- function(y,
                                    X,
                                    phi = 1,
                                    n_iter = 2000,
                                    burn_in = 0,
                                    level = 0.95,
                                    # spike-and-slab hyperparameters
                                    sigma0 = 1e-6,      # spike sd (very small)
                                    a_tau = 1, b_tau = 1, # IG prior for slab variance tau2 ~ Inv-Gamma(a_tau, b_tau)
                                    a_pi = 1, b_pi = 1) { # Beta prior for inclusion prob pi ~ Beta(a_pi, b_pi)
  # requirements
  if (!requireNamespace("BayesLogit", quietly = TRUE)) stop("Install BayesLogit")
  if (!requireNamespace("mvtnorm", quietly = TRUE)) stop("Install mvtnorm")
  
  n <- length(y)
  p <- ncol(X)
  tX <- t(X)
  eps <- 1e-10
  
  # storage
  beta_samples  <- matrix(0, nrow = n_iter, ncol = p)
  gamma_samples <- matrix(0, nrow = n_iter, ncol = p)
  tau2_samples  <- numeric(n_iter)
  pi_samples    <- numeric(n_iter)
  
  # init
  beta <- rep(0, p)
  gamma <- rep(1, p)      # start with all included
  tau2 <- 1.0             # slab variance
  pi <- 0.5               # prior inclusion prob
  
  for (iter in 1:n_iter) {
    ## 1) Polya-Gamma augmentation for Beta regression
    eta <- X %*% beta
    omega <- BayesLogit::rpg(n = n, h = phi, z = as.numeric(eta))
    
    ## 2) Update beta | omega, gamma, tau2  (Gaussian)
    # prior variance for beta_j is slab: tau2 if gamma_j=1, else sigma0^2
    prior_var <- ifelse(gamma == 1, tau2, sigma0^2)
    prior_prec <- 1 / pmax(prior_var, eps)
    
    Q <- tX %*% (omega * X) + diag(as.numeric(prior_prec), p, p)
    kappa <- phi * (y - 0.5)
    # robust solve / cholesky
    V_beta <- tryCatch({
      R <- chol(Q)
      chol2inv(R)
    }, error = function(e) {
      solve(Q + diag(eps, p))
    })
    m_beta <- as.numeric(V_beta %*% (tX %*% kappa))
    beta <- as.numeric(mvtnorm::rmvnorm(1, mean = m_beta, sigma = V_beta))
    
    ## 3) Update gamma_j | beta_j, pi  
    for (j in 1:p) {
      log_num <- log(pi + eps) + dnorm(beta[j], mean = 0, sd = sqrt(max(tau2, eps)), log = TRUE)
      log_den <- log(1 - pi + eps) + dnorm(beta[j], mean = 0, sd = sqrt(max(sigma0^2, eps)), log = TRUE)
      odds <- exp(log_num - log_den)
      prob_incl <- odds / (1 + odds)
      if (!is.finite(prob_incl)) prob_incl <- 0.5
      prob_incl <- min(max(prob_incl, 0), 1)  # clamp
      gamma[j] <- rbinom(1, 1, prob_incl)
    }
    
    ## 4) Update slab variance tau2 | beta, gamma
    sum_g <- sum(gamma, na.rm = TRUE)
    if (sum_g > 0) {
      a_post <- a_tau + 0.5 * sum_g
      b_post <- b_tau + 0.5 * sum((beta[gamma == 1])^2)
      tau2 <- 1 / rgamma(1, shape = a_post, rate = b_post + eps)
    } else {
      tau2 <- 1 / rgamma(1, shape = a_tau, rate = b_tau + eps)
    }
    if (!is.finite(tau2) || tau2 <= 0) tau2 <- 1
    
    
    ## 5) Update pi | gamma ~ Beta(a_pi + sum(gamma), b_pi + p - sum(gamma))
    pi <- rbeta(1, a_pi + sum_g, b_pi + p - sum_g)
    
    ## store
    beta_samples[iter, ] <- beta
    gamma_samples[iter, ] <- gamma
    tau2_samples[iter] <- tau2
    pi_samples[iter] <- pi
  }
  
  # posterior summaries after burn-in
  posterior_beta <- beta_samples[(burn_in + 1):n_iter, , drop = FALSE]
  posterior_gamma <- gamma_samples[(burn_in + 1):n_iter, , drop = FALSE]
  posterior_tau2 <- tau2_samples[(burn_in + 1):n_iter]
  posterior_pi <- pi_samples[(burn_in + 1):n_iter]
  
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
  ci <- credible_intervals(posterior_beta, level)
  # posterior inclusion probabilities (PIP)
  pip <- colMeans(posterior_gamma)
  selected_by_pip <- pip > 0.5
  
  return(list(
    beta_samples = posterior_beta,
    gamma_samples = posterior_gamma,
    tau2_samples = posterior_tau2,
    pi_samples = posterior_pi,
    pip = pip,
    selected_by_pip = selected_by_pip,
    ci = ci
  ))
}

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
library(tictoc)