#' Ordinal Logistic Regression with LASSO Penalty
#'
#' This function computes the log-likelihood value from the ordinal regression model (with a logit link) given data and parameters
#'
#' @param formula: a symbolic description of the model to be fitted; an object of class `"formula"`
#' @param data: a data frame containing the variables in the model
#' @param lambda: LASSO penalty parameter; a non-negative number; default is `lambda = 0`, which corresponds to no penalty term
#'
#' @return a list with the following elements:
#' \itemize{
#' \item{alpha: the estimated category specific intercepts for the lowest J-1 outcome categories; a numeric vector of length J-1}
#' \item{beta: the estimated slopes; a numeric vector of length p}
#' \item{vcov: asymptotic covariance matrix of unpenalized parameter estimates (NOTE: if lambda > 0, then this will overestimate the variance of the penalized estimates)}
#' \item{loglik: the log-likelihood value at convergence (not including the LASSO penalty)}
#' \item{lasso.penalty: the penalty term evaluated at the estimates for alpha and beta}
#' }
#'
#' @importFrom numDeriv hessian
#'
#' @export
ordreg.lasso <- function(formula, data, lambda = 0) {

  # extract ordered factor outcome vector from data
  Y <- data[, all.vars(formula)[1]]

  # number of observations
  n <- length(Y)

  # number of outcome categories
  J <- nlevels(Y)

  # convert factor levels to numeric values
  y <- numeric(n)
  for (j in 1:J) {
    y[Y == levels(Y)[j]] <- j
  }

  # extract covariate matrix (not including intercept)
  x <- model.matrix(formula, data)[, -1]

  # maximize likelihood
  opt.res <- optim(
    par = c(get.zeta(alpha = seq(.1, 1, length = J - 1)), # initial values for zeta
                           rep(0, ncol(x))),           # initial values for beta
    fn = function(theta) {
    beta <- theta[-(1:(J - 1))]
    zeta <- theta[1:(J - 1)]
    -(1 / n) * (
      # log-likelihood
      loglik(zeta = zeta,
             beta = beta,
             y = y,
             x = x)) +
      # LASSO penalty
      lambda * sum(abs(beta))
    },
  method = "BFGS")

  # estimated zeta, alpha, and beta
  zeta <- opt.res$par[1:(J - 1)]
  alpha <- get.alpha(zeta = zeta)
  beta <- opt.res$par[-(1:(J - 1))]

  # hessian matrix of log-likelihood (as a function of alpha and beta) at MLE
  hess <- numDeriv::hessian(
    func = function(theta) {
      beta <- theta[-(1:(J - 1))]
      zeta <- get.zeta(theta[1:(J - 1)])
      loglik(zeta = zeta,
             beta = beta,
             y = y,
             x = x)
    },
    x = c(alpha, beta)
  )

  # asymptotic covariance matrix by inverting hessian
  vcov <- -1 * solve(hess)

  # LASSO penalty term
  lasso.penalty <- lambda * sum(abs(beta))

  # log-likelihood value at convergence (not including LASSO penalty)
  ll <- -1 * length(y) * opt.res$value + lasso.penalty

  return(list("alpha" = alpha,
              "beta" = beta,
              "vcov" = vcov,
              "loglik" = ll,
              "lasso.penalty" = lasso.penalty))
}
