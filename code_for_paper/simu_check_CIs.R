library(BayesLogit);library(Rcpp);sourceCpp('test.cpp')   # For Polya-Gamma sampling
source('Gibbs_betreg_hs.R')
library(Rcpp); sourceCpp('lasso_betareg_.cpp')
library(Rcpp);sourceCpp('bayesbetareg_cpp2.cpp') 
library(Rcpp);sourceCpp('gibb_laplace_prior.cpp') 
library(Rcpp);sourceCpp('gibb_SnS_Gauss.cpp') 
phi <- 10
n_test <- 30
epsilon <- 1e-6
# --- Simulated Data ---
n <- 500
p <- 20   # predictors
s0 = 10
beta_true <- rep(0, p) ; beta_true[1:s0] <- c( rep(1, s0/2) , rep(-1, s0/2) )
rho <- 0.5; Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j)) ;LL = chol(Sigma) 


frequent_out = hs_out = dl_out = ss_out = list()

for (ii in 1:100) {
  xxx <- matrix(rnorm( (n + n_test)* p ), ncol = p )%*% LL
  eta <- xxx %*% beta_true
  mu <- 1 / (1 + exp(-eta)) ; 
  yyy <- rbeta(n + n_test, mu * phi, (1 - mu) * phi)
  yyy = pmin(pmax(yyy, epsilon), 1 - epsilon)
  X = xxx[1:n,]
  y = yyy[1:n]
  y_test = yyy[-(1:n) ]
  X_test = xxx[-(1:n),]
  
  sam_hs_cpp <- betareg_bayes_cpp(y,X,phi = 10,n_iter = 5000) 
  post_sams_hs <- sam_hs_cpp$beta_samples[(1000 + 1):5000, ]
  bhat_HScpp <- colMeans(post_sams_hs)
  ci_hs <- credible_intervals(post_sams_hs)
  CIhs_lower <- ci_hs[,'Lower'] 
  CIhs_upper <- ci_hs[,'Upper'] 
  CIhs_length <- CIhs_upper - CIhs_lower
  
  samples_DirLap <- betareg_bayes_dirlap_cpp(y,X,phi = 10,n_iter = 5000) 
  posterior_DirLap <- samples_DirLap$beta_samples[(1000 + 1):5000, ]
  bhat_DirLap <- colMeans(posterior_DirLap)
  ci_dl <- credible_intervals(posterior_DirLap)
  CIdl_lower <- ci_dl[,'Lower'] 
  CIdl_upper <- ci_dl[,'Upper'] 
  CIdl_length <- CIdl_upper - CIdl_lower
  
  samples_SnS_gaus <- betareg_SnS_gauss_cpp(y,X,phi = 10,n_iter = 5000) 
  posts_SnS_Gauss <- samples_SnS_gaus$beta_samples[(1000 + 1):5000, ]
  bhat_SnSgauss <- colMeans(posts_SnS_Gauss)
  ci_ss <- credible_intervals(posts_SnS_Gauss)
  CIss_lower <- ci_ss[,'Lower'] 
  CIss_upper <- ci_ss[,'Upper'] 
  CIss_length <- CIss_upper - CIss_lower
  
  df <- data.frame(y, X)
  # Fit the Beta regression model
  library(betareg)
  fit_betareg <- betareg(y ~ 0+ X, data = df)
  bereg <- fit_betareg$coefficients$mean  
  s_betareg <- summary(fit_betareg)$coefficients$mean[, "Std. Error"]
  # 95% confidence intervals
  CI_lower <- bereg - qnorm(0.975) * s_betareg
  CI_upper <- bereg + qnorm(0.975) * s_betareg
  CI <- cbind(CI_lower, CI_upper)
  CI_length <- CI_upper - CI_lower
  frequent_out[[ii]] = c( mean(CI_length) ,  mean(beta_true >= CI_lower & beta_true <= CI_upper) )
  
  hs_out[[ii]] =  c( mean(CIhs_length) , mean(beta_true >= CIhs_lower & beta_true <= CIhs_upper) )
  
  dl_out[[ii]] = c( mean(CIdl_length),mean(beta_true >= CIdl_lower & beta_true <= CIdl_upper) )
  
  ss_out[[ii]] = c(mean(CIhs_length),  mean(beta_true >= CIss_lower & beta_true <= CIss_upper) )
  
  print(ii)
}


for (jj in 1:2) {
  print( round( c(mean( unlist(sapply(frequent_out, function(x)x[jj])),na.rm=TRUE ), sd(unlist(sapply(frequent_out, function(x )x[jj]) ),na.rm=TRUE) ), 2) ) }
for (jj in 1:2) {
  print( round( c(mean( unlist(sapply(hs_out, function(x)x[jj])),na.rm=TRUE ), sd(unlist(sapply(hs_out, function(x )x[jj]) ),na.rm=TRUE) ), 2) ) }
for (jj in 1:2) {
  print( round( c(mean( unlist(sapply(dl_out, function(x)x[jj])),na.rm=TRUE ), sd(unlist(sapply(dl_out, function(x )x[jj]) ),na.rm=TRUE) ), 2) ) }
for (jj in 1:2) {
  print( round( c(mean( unlist(sapply(ss_out, function(x)x[jj])),na.rm=TRUE ), sd(unlist(sapply(ss_out, function(x )x[jj]) ),na.rm=TRUE) ), 2) ) }

