// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <Rcpp.h>
using namespace Rcpp;
using namespace arma;

// Mathematical constants (kept small set used)
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_PI_2      1.570796326794896619231321639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739

// Prototypes for Polya-Gamma helper functions (Windle)
double samplepg(double);
double exprnd(double);
double tinvgauss(double, double);
double truncgamma();
double randinvg(double);
double aterm(int, double, double);

// ---------- Polya-Gamma helpers (copied/adapted) ----------
NumericVector rcpp_pgdraw(NumericVector b, NumericVector c)
{
  int m = b.size();
  int n = c.size();
  NumericVector y(n);
  
  for (int i = 0; i < n; i++)
  {
    int bi = 1;
    if (m == 1)
      bi = std::max(1, (int)std::round(b[0]));
    else
      bi = std::max(1, (int)std::round(b[i]));
    
    double acc = 0.0;
    for (int j = 0; j < bi; j++) acc += samplepg(c[i]);
    y[i] = acc;
  }
  
  return y;
}

double samplepg(double z)
{
  if(!std::isfinite(z)) z = 0.0;
  double az = std::fabs(z);
  if (az < 1e-12) return 0.25 + R::rnorm(0.0, 1e-6);
  if (az > 20.0) {
    double omega_mean = 1.0 / (2.0 * az) * std::tanh(az/2.0);
    double omega_var = 1.0 / (4.0 * std::pow(az, 3)) * std::pow(1.0 / std::cosh(az/2.0), 2);
    if(!std::isfinite(omega_var) || omega_var <= 0) omega_var = 1e-8;
    return std::max(0.0, R::rnorm(omega_mean, std::sqrt(omega_var)));
  }
  
  double z0 = az * 0.5;
  double t = MATH_2_PI;
  double K = z0*z0/2.0 + MATH_PI2/8.0;
  double logA = std::log(4.0) - MATH_LOG_PI - z0;
  double logK = std::log(K);
  double Kt = K * t;
  double w = std::sqrt(MATH_PI_2);
  
  double logf1 = logA + R::pnorm(w*(t*z0 - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z0 + R::pnorm(-w*(t*z0+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = std::exp(logf1) + std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q);
  
  double u, X;
  while(1)
  {
    u = R::runif(0.0,1.0);
    if(u < ratio) X = t + exprnd(1.0)/K;
    else X = tinvgauss(z0, t);
    
    int i = 1;
    double Sn = aterm(0, X, t);
    double U = R::runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;
    
    while(1)
    {
      Sn = Sn + asgn * aterm(i, X, t);
      if(!even && (U <= Sn)) { X = X * 0.25; return X; }
      if(even && (U > Sn)) break;
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  return 0.25 * X;
}

double exprnd(double mu) {
  double u = R::runif(0.0,1.0);
  if(u >= 1.0) u = std::nextafter(1.0, 0.0);
  return -mu * std::log(1.0 - u);
}

double aterm(int n, double x, double t)
{
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }
  return std::exp(f);
}

double randinvg(double mu)
{
  double u = R::rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - std::sqrt(4.0*mu*V + mu*mu * V*V) );
  if(R::runif(0.0,1.0) > mu /(mu+out)) out = mu*mu / out;
  return out;
}

double truncgamma() {
  double c = MATH_PI_2;
  double X, gX;
  bool done = false;
  while(!done)
  {
    X = exprnd(1.0) * 2.0 + c;
    gX = 1.2533141373155002512078826424055 / std::sqrt(X); // sqrt(pi/2)
    if(R::runif(0.0,1.0) <= gX) done = true;
  }
  return X;
}

double tinvgauss(double z, double t) {
  double X, u;
  double mu = 1.0/z;
  if(mu > t) {
    while(1) {
      u = R::runif(0.0, 1.0);
      X = 1.0 / truncgamma();
      if (std::log(u) < (-z*z*0.5*X)) break;
    }
  }
  else {
    X = t + 1.0;
    while(X >= t) X = randinvg(mu);
  }
  return X;
}

// multivariate normal via precision-cholesky solve (no explicit inverse)
arma::vec rmvnorm_prec_chol(const arma::mat &Q, const arma::vec &bvec) {
  int p = Q.n_rows;
  arma::mat R;
  try {
    R = arma::chol(Q); // upper triangular, R.t()*R = Q
  } catch (...) {
    arma::mat V = arma::pinv(Q);
    arma::vec mu = V * bvec;
    return mu + arma::chol(V, "lower") * arma::randn(p);
  }
  
  // solve for mean: Q^{-1} bvec
  arma::vec u = arma::solve(arma::trimatl(R.t()), bvec);
  arma::vec mu = arma::solve(arma::trimatu(R), u);
  
  // sample N(0, Q^{-1}) by solving R v = z
  arma::vec z = arma::randn(p);
  arma::vec v = arma::solve(arma::trimatu(R), z);
  return mu + v;
}

// [[Rcpp::export]]
Rcpp::List betareg_SnS_gauss_cpp(const arma::vec &y,
                                 const arma::mat &X,
                                 double phi = 1.0,
                                 int n_iter = 2000,
                                 int burn_in = 0,
                                 double level = 0.95,
                                 double s0 = 0.01,    // spike SD (small)
                                 double s1 = 1.0,     // slab SD (large)
                                 double a_pi = 1.0, double b_pi = 1.0) {
  if (burn_in < 0) burn_in = 0;
  if (burn_in >= n_iter) Rcpp::stop("burn_in must be < n_iter");
  int n = y.n_elem;
  int p = X.n_cols;
  double eps = 1e-12;
  
  // Storage
  arma::mat beta_samples(n_iter, p, fill::zeros);
  arma::mat gamma_samples(n_iter, p, fill::zeros);
  arma::mat priorvar_samples(n_iter, p, fill::zeros);
  arma::vec pi_samples(n_iter, fill::zeros);
  
  // Initialize
  arma::vec beta(p, fill::zeros);
  arma::uvec gamma(p, fill::zeros);
  arma::vec prior_var(p, fill::value(s1*s1)); // start with slab var
  double pi = 0.5;
  
  arma::mat tX = X.t();
  
  // integer phi for PG sampler (counts)
  //int phi_int = std::max(1, (int)std::round(phi));
  //if(std::abs(phi - phi_int) > 1e-8) Rcpp::Rcout << "Note: phi rounded to integer "<< phi_int <<" for PG augmentation.\n";
  
  // Precompute log normalization constants for Gaussian densities
  double log_norm_s0 = -0.5 * std::log(2.0 * M_PI * s0 * s0);
  double log_norm_s1 = -0.5 * std::log(2.0 * M_PI * s1 * s1);
  
  for (int iter = 0; iter < n_iter; iter++) {
    // 1) Polya-Gamma augmentation
    arma::vec eta = X * beta;
    //NumericVector b_pg(n); b_pg.fill(static_cast<double>(phi_int));
    NumericVector b_pg(n, phi);
    NumericVector c_pg(eta.begin(), eta.end());
    NumericVector omega_r = rcpp_pgdraw(b_pg, c_pg);
    arma::vec omega = as<arma::vec>(omega_r);
    
    // 2) Update beta | omega, prior_var (Gaussian prior with var = prior_var_j)
    arma::vec prior_prec = 1.0 / arma::max(prior_var, arma::vec(p, fill::value(eps)));
    arma::mat Q = tX * (X.each_col() % omega) + arma::diagmat(prior_prec);
    arma::vec kappa = phi * (y - 0.5);
    arma::vec bvec = tX * kappa;
    
    beta = rmvnorm_prec_chol(Q, bvec);
    
    // 3) Update gamma_j | beta_j (Gaussian mixture)
    for (int j = 0; j < p; j++) {
      double bj = beta(j);
      double log_num = std::log(pi + eps) + R::dnorm(bj, 0.0, std::max(s1, eps), 1); // log density
      double log_den = std::log(1.0 - pi + eps) + R::dnorm(bj, 0.0, std::max(s0, eps), 1); // log density
      double logit = log_num - log_den;
      double p_incl;
      if (logit >= 0) p_incl = 1.0 / (1.0 + std::exp(-logit));
      else { double e = std::exp(logit); p_incl = e / (1.0 + e); }
      double u = R::runif(0.0,1.0);
      gamma(j) = (u < p_incl) ? 1 : 0;
      
      // set prior variance accordingly (store variance)
      prior_var(j) = (gamma(j) == 1) ? (s1*s1) : (s0*s0);
    }
    
    // 4) Update pi | gamma ~ Beta
    int sum_g = arma::sum(gamma);
    double ap_post = a_pi + sum_g;
    double bp_post = b_pi + p - sum_g;
    if (ap_post <= 0) ap_post = eps;
    if (bp_post <= 0) bp_post = eps;
    pi = R::rbeta(ap_post, bp_post);
    
    // 5) Store
    beta_samples.row(iter) = beta.t();
    for (int j = 0; j < p; j++) gamma_samples(iter, j) = (double)gamma(j);
    priorvar_samples.row(iter) = prior_var.t();
    pi_samples(iter) = pi;
    
    // Optional diagnostics (uncomment to see)
    // if(iter < 20 || iter % 1000 == 0) {
    //   Rcpp::Rcout << "iter="<<iter<<" sum(gamma)="<<arma::sum(gamma)
    //               <<" max|beta|="<<arma::max(arma::abs(beta))
    //               <<" mean(omega)="<<arma::mean(omega)<<" pi="<<pi<<"\n";
    // }
  }
  
  // Posterior summaries
  arma::mat posterior_beta = beta_samples.rows(burn_in, n_iter - 1);
  arma::mat posterior_gamma = gamma_samples.rows(burn_in, n_iter - 1);
  arma::mat posterior_priorvar = priorvar_samples.rows(burn_in, n_iter - 1);
  arma::vec posterior_pi = pi_samples.subvec(burn_in, n_iter - 1);
  
  arma::mat ci(p, 3);
  arma::vec selected(p, fill::zeros);
  double alpha = (1.0 - level) / 2.0;
  for (int j = 0; j < p; j++) {
    arma::vec col = arma::sort(posterior_beta.col(j));
    int n_keep = col.n_elem;
    int ilow = std::max(0, (int) std::floor(alpha * n_keep));
    int ihigh = std::min(n_keep - 1, (int) std::floor((1 - alpha) * n_keep));
    int imed = std::max(0, (int) std::floor(0.5 * n_keep));
    ci(j, 0) = col(ilow);
    ci(j, 1) = col(imed);
    ci(j, 2) = col(ihigh);
    if (!(ci(j,0) < 0.0 && ci(j,2) > 0.0)) selected(j) = 1.0;
  }
  
  arma::rowvec pip = arma::mean(posterior_gamma, 0);
  arma::uvec selected_by_pip(p);
  for (int j = 0; j < p; j++) selected_by_pip(j) = (pip(j) > 0.5) ? 1 : 0;
  
  return List::create(
    Named("beta_samples") = posterior_beta,
    Named("gamma_samples") = posterior_gamma,
    Named("priorvar_samples") = posterior_priorvar,
    Named("pi_samples") = posterior_pi,
    Named("pip") = pip.t(),
    Named("selected_by_pip") = selected_by_pip,
    Named("ci") = ci
  );
}
