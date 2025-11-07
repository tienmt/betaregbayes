#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// Inverse logit with clipping
inline double inv_logit(double eta) {
  double val = 1.0 / (1.0 + std::exp(-eta));
  if (val < 1e-6) return 1e-6;
  if (val > 1 - 1e-6) return 1 - 1e-6;
  return val;
}

// Soft-thresholding operator
inline double soft_threshold(double z, double gamma) {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0.0;
}

// Log-likelihood for Beta regression
double loglik_beta(const vec& y, const vec& mu, double phi) {
  int n = y.n_elem;
  double ll = 0.0;
  for (int i = 0; i < n; i++) {
    double a = mu[i] * phi;
    double b = (1.0 - mu[i]) * phi;
    if (a <= 0 || b <= 0) return -std::numeric_limits<double>::infinity();
    ll += R::dbeta(y[i], a, b, true);
  }
  return ll;
}

// Estimate phi by maximizing log-likelihood (grid search)

double estimate_phi(const vec& y, const vec& mu,
                    double phi_start = 10.0,
                    double low_phi = 0.1,
                    double up_phi = 100.0,
                    int max_iter = 50,
                    double lambda_phi = 0.05,
                    double phi0 = 10.0) {
  double best_phi = phi_start;
  double best_val = -std::numeric_limits<double>::infinity();
  
  for (int iter = 0; iter < max_iter; iter++) {
    double phi = low_phi + (up_phi - low_phi) * iter / (max_iter - 1.0);
    double val = loglik_beta(y, mu, phi);
    
    // Quadratic penalty
    double penalty = lambda_phi * std::pow(phi - phi0, 2) / 2.0;
    double penalized_val = val - penalty;
    
    if (penalized_val > best_val) {
      best_val = penalized_val;
      best_phi = phi;
    }
  }
  return best_phi;
}


// [[Rcpp::export]]
List lasso_beta_phi_cpp(const mat& X, const vec& y,
                        double lambda,
                        double phi_init = 2.0,
                        int max_iter = 100,
                        double tol = 1e-6) {
  
  int n = X.n_rows;
  int p = X.n_cols;
  
  vec beta(p, fill::zeros);
  double phi = phi_init;
  
  double prev_ll = -std::numeric_limits<double>::infinity();
  
  for (int outer = 0; outer < max_iter; outer++) {
    vec eta = X * beta;
    vec mu(n);
    for (int i = 0; i < n; i++) mu[i] = inv_logit(eta[i]);
    
    vec W = phi * mu % (1 - mu);
    vec z = eta + (y - mu) / (mu % (1 - mu));
    
    vec r = z - X * beta; // initial residual
    for (int j = 0; j < p; j++) {
      double zj = dot(W % X.col(j), r + X.col(j) * beta[j]);
      double Xj_sq = dot(W, square(X.col(j)));
      double beta_new = soft_threshold(zj / Xj_sq, lambda / Xj_sq);
      
      double diff = beta_new - beta[j];
      if (diff != 0) {
        r -= diff * X.col(j);   // O(n) update
        beta[j] = beta_new;
      }
      if (std::abs(beta[j]) < 1e-4) beta[j] = 0.0;
      
    }
    
    // Coordinate descent for beta
    //for (int j = 0; j < p; j++) {
    //  vec r_j = z - X * beta + X.col(j) * beta[j];
    //  double z_j = dot(W % X.col(j), r_j);
    //  double Xj_sq = dot(W, square(X.col(j)));
    //  beta[j] = soft_threshold(z_j / Xj_sq, lambda / Xj_sq);
    //}
    
    // Update phi
    eta = X * beta;
    for (int i = 0; i < n; i++) mu[i] = inv_logit(eta[i]);
    double phi_new = estimate_phi(y, mu, phi);
    
    // Compute log-likelihood
    double curr_ll = loglik_beta(y, mu, phi_new);
    
    // Convergence check using log-likelihood
    if (std::abs(curr_ll - prev_ll) < tol) {
      phi = phi_new;
      break;
    }
    phi = phi_new;
    prev_ll = curr_ll;
  }
  
  return List::create(Named("beta") = NumericVector(beta.begin(), beta.end()),
                      Named("phi") = phi);
}

// [[Rcpp::export]]
List cv_lasso_beta_phi_cpp(const mat& X, const vec& y,
                           const vec& lambdas,
                           int K = 5,
                           double phi_init = 2.0,
                           int max_iter = 100,
                           double tol = 1e-6) {
  
  int n = X.n_rows;
  int n_lambda = lambdas.n_elem;
  vec cv_errors(n_lambda, fill::zeros);
  
  // --- Create fold assignment (0..K-1) ---
  std::vector<int> folds(n);
  for (int i = 0; i < n; i++) folds[i] = i % K;
  
  // shuffle with modern C++ engine
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(folds.begin(), folds.end(), g);
  
  // --- Loop over lambdas ---
  for (int i = 0; i < n_lambda; i++) {
    double lambda = lambdas[i];
    vec errors(K, fill::zeros);
    
    for (int k = 0; k < K; k++) {
      // Training/test split
      std::vector<uword> train_idx, test_idx;
      for (int j = 0; j < n; j++) {
        if (folds[j] == k) test_idx.push_back(j);
        else train_idx.push_back(j);
      }
      
      uvec train = conv_to<uvec>::from(train_idx);
      uvec test  = conv_to<uvec>::from(test_idx);
      
      mat X_train = X.rows(train);
      vec y_train = y.elem(train);
      mat X_test  = X.rows(test);
      vec y_test  = y.elem(test);
      
      // Fit model
      List fit = lasso_beta_phi_cpp(X_train, y_train, lambda,
                                    phi_init, max_iter, tol);
      vec beta_hat = fit["beta"];
      double phi_hat = as<double>(fit["phi"]);
      
      // Predict
      vec eta_test = X_test * beta_hat;
      vec mu_pred(eta_test.n_elem);
      for (uword t = 0; t < eta_test.n_elem; t++)
        mu_pred[t] = inv_logit(eta_test[t]);
      
      // Negative log-likelihood
      //double ll = loglik_beta(y_test, mu_pred, phi_hat);
      double ll = -mean(square(y_test - mu_pred));
      errors[k] = -ll;
    }
    
    cv_errors[i] = mean(errors);
  }
  
  // Best lambda
  uword best_idx = cv_errors.index_min();
  double best_lambda = lambdas[best_idx];
  
  return List::create(Named("best_lambda") = best_lambda,
                      Named("cv_errors") = cv_errors);
}
