
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rmath.h>


//#include <omp.h>

#include <Rcpp.h>
using namespace Rcpp;
using namespace arma;

// Mathematical constants computed using Wolfram Alpha
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

// FCN prototypes
double samplepg(double);
double exprnd(double);
double tinvgauss(double, double);
double truncgamma();
double randinvg(double);
double aterm(int, double, double);


NumericVector rcpp_pgdraw(NumericVector b, NumericVector c)
{
  int m = b.size();
  int n = c.size();
  NumericVector y(n);
  
  // Setup
  int i, j, bi = 1;
  if (m == 1)
  {
    bi = b[0];
  }
  
  // Sample
  for (i = 0; i < n; i++)
  {
    if (m > 1)
    {
      bi = b[i];
    }
    
    // Sample
    y[i] = 0;
    for (j = 0; j < (int)bi; j++)
    {
      y[i] += samplepg(c[i]);
    }
  }
  
  return y;
}


// Sample PG(1,z) with hybrid approximation
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z)
{
  // --- Hybrid approximation for large |z| ---
  if (std::fabs(z) > 20.0) {  // threshold can be tuned
    double omega_mean = 1.0 / (2.0 * std::fabs(z)) * std::tanh(std::fabs(z)/2.0);
    return omega_mean;         // deterministic mean
    // Alternatively, add small noise:
    // double omega_var = 1.0 / (4.0 * std::pow(std::fabs(z), 3)) * std::pow(1.0 / std::cosh(z/2.0), 2);
    // return R::rnorm(omega_mean, std::sqrt(omega_var));
  }
  
  // --- Original Windle sampler for moderate |z| ---
  z = std::fabs(z) * 0.5;  // PG(b,z) = 0.25 * J*(b, z/2)
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
  
  double u, X;
  
  while(1) 
  {
    u = R::runif(0.0,1.0);
    if(u < ratio) {
      X = t + exprnd(1.0)/K;
    }
    else {
      X = tinvgauss(z, t);
    }
    
    int i = 1;
    double Sn = aterm(0, X, t);
    double U = R::runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;
    
    while(1) 
    {
      Sn = Sn + asgn * aterm(i, X, t);
      
      if(!even && (U <= Sn)) {
        X = X * 0.25;
        return X;
      }
      
      if(even && (U > Sn)) {
        break;
      }
      
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  
  return X;
}


// Generate exponential distribution random variates
double exprnd(double mu)
{
  return -mu * (double)std::log(1.0 - (double)R::runif(0.0,1.0));
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t)
{
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }    
  return (double)exp(f);
}

// Generate inverse gaussian random variates
double randinvg(double mu)
{
  // sampling
  double u = R::rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - (double)std::sqrt(4.0*mu*V + mu*mu * V*V) );
  
  if(R::runif(0.0,1.0) > mu /(mu+out)) {    
    out = mu*mu / out; 
  }    
  return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma()
{
  double c = MATH_PI_2;
  double X, gX;
  
  bool done = false;
  while(!done)
  {
    X = exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / (double)std::sqrt(X);
    
    if(R::runif(0.0,1.0) <= gX) {
      done = true;
    }
  }
  
  return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t)
{
  double X, u;
  double mu = 1.0/z;
  
  // Pick sampler
  if(mu > t) {
    // Sampler based on truncated gamma 
    // Algorithm 3 in the Windle (2013) PhD thesis, page 128
    while(1) {
      u = R::runif(0.0, 1.0);
      X = 1.0 / truncgamma();
      
      if ((double)std::log(u) < (-z*z*0.5*X)) {
        break;
      }
    }
  }  
  else {
    // Rejection sampler
    X = t + 1.0;
    while(X >= t) {
      X = randinvg(mu);
    }
  }    
  return X;
}
// Helper: sample from inverse-Gaussian
double rinvgauss(double mu, double lambda) {
  double v = R::rnorm(0,1);
  double y = v*v;
  double x = mu + (mu*mu*y)/(2*lambda) - (mu/(2*lambda)) * sqrt(4*mu*lambda*y + mu*mu*y*y);
  if (R::runif(0,1) <= mu / (mu + x)) return x;
  else return mu*mu / x;
}

// Helper: multivariate normal sampler
arma::vec rmvnorm(const arma::vec &mean, const arma::mat &Sigma) {
  arma::vec z = arma::randn(mean.n_elem);
  arma::mat L = arma::chol(Sigma, "lower");
  return mean + L*z;
}

// [[Rcpp::export]]
Rcpp::List betareg_bayes_dirlap_cpp(const arma::vec &y,
                                    const arma::mat &X,
                                    double phi = 1.0,
                                    int n_iter = 1000,
                                    int burn_in = 0,
                                    double level = 0.95,
                                    double a = 0.0) {
  Rcpp::Function rgig("rgig", Rcpp::Environment::namespace_env("GIGrvg"));
  
  //Function rgig("GIGrvg::rgig"); // call R's GIG sampler
  int n = y.n_elem;
  int p = X.n_cols;
  if (a <= 0.0) a = 1.0 / p;
  
  // Storage
  arma::mat beta_samples(n_iter, p, fill::zeros);
  arma::mat delta_samples(n_iter, p, fill::zeros);
  arma::mat psi_samples(n_iter, p, fill::zeros);
  
  // Initialize
  arma::vec beta(p, fill::zeros);
  arma::vec delta(p, fill::ones);
  arma::vec psi(p, fill::ones);
  double eps = 1e-10;
  
  arma::mat tX = X.t();
  
  for (int iter = 0; iter < n_iter; iter++) {
    // ----------------------------
    // 1) Polya-Gamma augmentation
    // ----------------------------
    arma::vec eta = X * beta; // n-vector
    
    Rcpp::NumericVector eta_r(eta.begin(), eta.end());
    Rcpp::NumericVector b(n, phi);   // vector of shape params
    Rcpp::NumericVector c(eta.begin(), eta.end()); // tilt params
    Rcpp::NumericVector omega_r = rcpp_pgdraw(b, c);
    
    arma::vec omega = as<arma::vec>(omega_r);
    
    // ----------------------------
    // 2) Update beta | omega, delta, psi
    // ----------------------------
    arma::vec prior_prec = 1.0 / arma::max(psi % arma::square(delta), arma::vec(p, fill::value(eps)));
    arma::mat Q = tX * (X.each_col() % omega) + arma::diagmat(prior_prec);
    arma::vec kappa = phi * (y - 0.5);
    
    arma::mat V_beta;
    arma::vec m_beta;
    try {
      arma::mat R = arma::chol(Q);
      V_beta = arma::inv(arma::trimatu(R)) * arma::inv(arma::trimatu(R)).t();
      m_beta = V_beta * (tX * kappa);
    } catch (...) {
      V_beta = arma::pinv(Q);
      m_beta = V_beta * (tX * kappa);
    }
    beta = rmvnorm(m_beta, V_beta);
    
    // ----------------------------
    // 3) Dirichlet-Laplace updates
    // ----------------------------
    for (int j = 0; j < p; j++) {
      double chi_j = 2.0 * std::max(std::abs(beta(j)), eps);
      NumericVector delta_j_vec = rgig(1, a - 1.0, chi_j, 1.0);
      double delta_j = std::max(delta_j_vec[0], 1e-6);
      delta(j) = delta_j;
      
      double mu_ig = delta(j) / std::max(std::abs(beta(j)), eps);
      double psi_j = 1.0 / rinvgauss(mu_ig, 1.0);
      psi(j) = std::max(psi_j, 1e-6);
    }
    
    // ----------------------------
    // 4) Store
    // ----------------------------
    beta_samples.row(iter) = beta.t();
    delta_samples.row(iter) = delta.t();
    psi_samples.row(iter) = psi.t();
  }
  
  // ----------------------------
  // Posterior summaries
  // ----------------------------
  arma::mat posterior_beta = beta_samples.rows(burn_in, n_iter - 1);
  
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
  
  return List::create(
    Named("beta_samples") = posterior_beta,
    Named("delta_samples") = delta_samples.rows(burn_in, n_iter - 1),
    Named("psi_samples") = psi_samples.rows(burn_in, n_iter - 1),
    Named("selected_variable") = selected,
    Named("ci") = ci
  );
}