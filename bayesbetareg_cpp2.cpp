// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <Rmath.h>
#include <Rcpp.h>
using namespace Rcpp;
using namespace RcppParallel;

// Mathematical constants
#define MATH_PI        3.14159265358979323846
#define MATH_PI_2      1.57079632679489661923
#define MATH_2_PI      0.63661977236758134308
#define MATH_PI2       9.86960440108935861883
#define MATH_PI2_2     4.93480220054467930941
#define MATH_SQRT1_2   0.70710678118654752440
#define MATH_SQRT_PI_2 1.25331413731550025121
#define MATH_LOG_PI    1.14472988584940017414
#define MATH_LOG_2_PI  -0.45158270528945486473
#define MATH_LOG_PI_2  0.45158270528945486473

// --- Function prototypes ---
double samplepg(double);
double exprnd(double);
double tinvgauss(double, double);
double truncgamma();
double randinvg(double);
double aterm(int, double, double);


// Worker struct
struct PGDrawWorker : public Worker {
  const RVector<double> b;
  const RVector<double> c;
  RVector<double> y;
  int m;
  
  PGDrawWorker(const NumericVector& b, const NumericVector& c, NumericVector& y)
    : b(b), c(c), y(y), m(b.size()) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      int bi = (m == 1) ? static_cast<int>(b[0]) : static_cast<int>(b[i]);
      double sum_pg = 0.0;
      
      for (int j = 0; j < bi; j++) {
        sum_pg += samplepg(c[i]);
      }
      y[i] = sum_pg;
    }
  }
};

// [[Rcpp::export]]
NumericVector rcpp_pgdraw_parallel(const NumericVector& b, const NumericVector& c) {
  int n = c.size();
  NumericVector y(n);
  
  PGDrawWorker worker(b, c, y);
  parallelFor(0, n, worker);
  
  return y;
}

// [[Rcpp::export]]
NumericVector rcpp_pgdraw(NumericVector b, NumericVector c) {
  int m = b.size();
  int n = c.size();
  NumericVector y(n);
  
  int bi = 1;
  if (m == 1) {
    bi = b[0];
  }
  
  for (int i = 0; i < n; i++) {
    if (m > 1) {
      bi = b[i];
    }
    y[i] = 0;
    for (int j = 0; j < bi; j++) {
      y[i] += samplepg(c[i]);
    }
  }
  return y;
}

// ---------------- PG sampler utilities ----------------

// PG(1, z)
double samplepg(double z) {
  if (std::fabs(z) > 20.0) {
    double omega_mean = 1.0 / (2.0 * std::fabs(z)) * std::tanh(std::fabs(z)/2.0);
    return omega_mean;
  }
  z = std::fabs(z) * 0.5;
  double t = MATH_2_PI;
  
  double K = z*z/2.0 + MATH_PI2/8.0;
  double logA = std::log(4.0) - MATH_LOG_PI - z;
  double logK = std::log(K);
  double Kt = K * t;
  double w = std::sqrt(MATH_PI_2);
  
  double logf1 = logA + R::pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z + R::pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = std::exp(logf1) + std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q); 
  
  while (true) {
    double u = R::runif(0.0,1.0);
    double X;
    if (u < ratio) {
      X = t + exprnd(1.0)/K;
    } else {
      X = tinvgauss(z, t);
    }
    
    int i = 1;
    double Sn = aterm(0, X, t);
    double U = R::runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;
    
    while (true) {
      Sn = Sn + asgn * aterm(i, X, t);
      if (!even && (U <= Sn)) {
        return X * 0.25;
      }
      if (even && (U > Sn)) {
        break;
      }
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
}

// Exp(mu)
double exprnd(double mu) {
  return -mu * std::log(1.0 - R::runif(0.0,1.0));
}

// a-term
double aterm(int n, double x, double t) {
  double f = 0.0;
  if (x <= t) {
    f = MATH_LOG_PI + std::log(n + 0.5) +
      1.5*(MATH_LOG_2_PI - std::log(x)) -
      2*(n + 0.5)*(n + 0.5)/x;
  } else {
    f = MATH_LOG_PI + std::log(n + 0.5) -
      x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }
  return std::exp(f);
}

// Inv-Gaussian
double randinvg(double mu) {
  double u = R::rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - std::sqrt(4.0*mu*V + mu*mu * V*V) );
  if (R::runif(0.0,1.0) > mu /(mu+out)) {
    out = mu*mu / out; 
  }
  return out;
}

// Truncated gamma
double truncgamma() {
  double c = MATH_PI_2;
  double X, gX;
  while (true) {
    X = exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / std::sqrt(X);
    if (R::runif(0.0,1.0) <= gX) break;
  }
  return X;  
}

// Truncated Inv-Gaussian
double tinvgauss(double z, double t) {
  double mu = 1.0/z;
  double X;
  if (mu > t) {
    while (true) {
      double u = R::runif(0.0, 1.0);
      X = 1.0 / truncgamma();
      if (std::log(u) < (-z*z*0.5*X)) break;
    }
  } else {
    X = t + 1.0;
    while (X >= t) {
      X = randinvg(mu);
    }
  }
  return X;
}

// MVN sampler
arma::vec rmvnorm(const arma::vec &m, const arma::mat &V) {
  arma::vec z = arma::randn<arma::vec>(m.n_elem);
  arma::mat L = arma::chol(V, "lower");
  return m + L * z;
}

// [[Rcpp::export]]
Rcpp::List betareg_bayes_cpp(const arma::vec &y,
                             const arma::mat &X,
                             double phi = 1.0,
                             double alpha_fractional = 1.0,
                             int n_iter = 1000,
                             int burn_in = 0,
                             double level = 0.95) {
  int n = y.n_elem;
  int p = X.n_cols;
  arma::mat tX = X.t();
  
  arma::mat beta_samples(n_iter, p, arma::fill::zeros);
  arma::vec tau_sanple(n_iter, arma::fill::zeros);
  arma::vec beta(p, arma::fill::zeros);
  arma::vec lambda2(p, arma::fill::ones);
  arma::vec nu(p, arma::fill::ones);
  double tau2 = 1.0;
  double xi = 1.0;
  
  for (int iter = 0; iter < n_iter; iter++) {
    arma::vec eta = X * beta;
    
    NumericVector b(n, alpha_fractional * phi);
    NumericVector c(eta.begin(), eta.end());
    NumericVector omega_r = rcpp_pgdraw_parallel(b, c);
    arma::vec omega = as<arma::vec>(omega_r);
    
    arma::vec kappa = alpha_fractional * phi * (y - 0.5);
    arma::mat Lambda_inv = arma::diagmat(1.0 / (lambda2 * tau2));
    
    // Build A = X' W X + Lambda_inv (A is p x p)
    arma::mat XtWX = tX * (X.each_col() % omega); // tX*(X % omega)
    arma::mat A = XtWX; 
    // add diagonal contributions from Lambda_inv = diag(1/(lambda2 * tau2))
    for (int j = 0; j < p; ++j) A(j,j) += 1.0 / (lambda2(j) * tau2);
    
    // compute m_beta solving A * m_beta = tX * kappa
    arma::vec rhs = tX * kappa;
    
    // Cholesky (lower)
    arma::mat L;
    bool ok = arma::chol(L, A, "lower"); // A = L * L.t()
    if (!ok) Rcpp::stop("Cholesky failed");
    
    // solve for m_beta: A m = rhs => L * L.t() * m = rhs -> solve L * (L.t()*m) = rhs
    arma::vec y = arma::solve(arma::trimatl(L), rhs);
    arma::vec m_beta = arma::solve(arma::trimatu(L.t()), y);
    // sample z ~ N(0, I)
    arma::vec z = arma::randn<arma::vec>(p);
    // compute sample = m_beta + L^{-T} z
    arma::vec u = arma::solve(arma::trimatu(L.t()), z);
    beta = m_beta + u;
    
    for (int j = 0; j < p; j++) {
      double rate1 = 1.0 / nu(j) + std::pow(beta(j), 2) / (2.0 * tau2);
      lambda2(j) = 1.0 / R::rgamma(1.0, 1.0 / rate1);
      double rate2 = 1.0 + 1.0 / lambda2(j);
      nu(j) = 1.0 / R::rgamma(1.0, 1.0 / rate2);
    }
    
    double rate_tau = 1.0 / xi + arma::accu(arma::square(beta) / lambda2) / 2.0;
    tau2 = 1.0 / R::rgamma((p + 1.0) / 2.0, 1.0 / rate_tau);
    double rate_xi = 1.0 + 1.0 / tau2;
    xi = 1.0 / R::rgamma(1.0, 1.0 / rate_xi);
    
    beta_samples.row(iter) = beta.t();
    tau_sanple(iter) = tau2;
  }
  
  arma::mat posterior_samples = beta_samples.rows(burn_in, n_iter - 1);
  
  arma::mat ci(p, 3);
  arma::vec selected(p, arma::fill::zeros);
  double alpha = (1.0 - level) / 2.0;
  
  for (int j = 0; j < p; j++) {
    arma::vec col = arma::sort(posterior_samples.col(j));
    int n_keep = col.n_elem;
    int ilow = std::max(0, (int)std::floor(alpha * n_keep));
    int ihigh = std::min(n_keep - 1, (int)std::floor((1.0 - alpha) * n_keep));
    int imed = std::max(0, (int)std::floor(0.5 * n_keep));
    double lower = col(ilow);
    double median = col(imed);
    double upper = col(ihigh);
    ci(j, 0) = lower; ci(j, 1) = median; ci(j, 2) = upper;
    if (!(lower < 0.0 && upper > 0.0)) selected(j) = 1.0;
  }
  
  return Rcpp::List::create(
    _["beta_samples"] = posterior_samples,
    _["selected_variable"] = selected,
    _["tau_sample"] = tau_sanple,
    _["ci"] = ci
  );
}
