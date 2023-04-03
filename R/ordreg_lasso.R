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
#' \item{loglik: the log-likelihood value at convergence (not including the LASSO penalty)}
#' \item{lasso.penalty: the penalty term evaluated at the estimates for alpha and beta}
#' }
#'
#' @export
ordreg.lasso <- function(formula, data, lambda = 0) {

  # extract outcome vector from data
  y <- data[, all.vars(formula)[1]]

  # number of categories minus 1
  J_1 <- nlevels(y) - 1

  # extract covariate matrix (not including intercept)
  x <- model.matrix(formula, data)[, -1]

  # maximize likelihood
  opt.res <- optim(par = c(seq(0, 1, length = J_1),
                           rep(0, ncol(x))),
                   fn = function(theta) {
                     beta <- theta[-(1:J_1)]
                     zeta <- theta[1:J_1]
                     -(1 / length(y)) * (
                       # log-likelihood
                       loglik(zeta = zeta,
                              beta = beta,
                              y = y,
                              x = x)) +
                       # LASSO penalty
                       lambda * sum(abs(beta))
                   },
                   method = "BFGS")
                   #method = "Nelder-Mead")

  # estimated alpha and beta
  alpha <- cumsum(exp(opt.res$par[1:J_1]))
  beta <- opt.res$par[-(1:J_1)]

  # LASSO penalty term
  lasso.penalty <- lambda * sum(abs(beta))

  # log-likelihood value at convergence (not including LASSO penalty)
  ll <- -1 * length(y) * opt.res$value + lasso.penalty

  return(list("alpha" = alpha,
              "beta" = beta,
              "loglik" = ll,
              "lasso.penalty" = lasso.penalty))
}
