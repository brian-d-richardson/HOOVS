#' Ordinal Logistic Regression Log-likelihood Function
#'
#' This function computes the log-likelihood value from the ordinal regression model (with a logit link) given data and parameters
#'
#' @param y: outcome vector; a numeric vector of length n with values in 1, ..., J corresponding to the J ordinal outcome categories
#' @param x: covariate matrix; n x p matrix with numeric elements
#' @param zeta: used to generate category specific intercepts alpha = sum(exp(zeta)); numeric vector of length J-1
#' @param beta: slope parameters; numeric vector of length p
#'
#' @return the log-likelihood value
#'
#' @export
loglik <- function(y, x, zeta, beta) {

  # number of observations
  n <- length(y)

  # number of ordered factor levels
  J <- max(y)

  # category-specific intercepts
  alpha = get.alpha(zeta)

  # linear predictor: a vector of length n
  eta <- x %*% beta

  # for each observation y, compute P(Y <= y | xi) - P(Y <= y - 1 | xi)
  probs <- numeric(n)
  for (i in 1:n) {
    probs[i] <- ifelse(y[i] == J, 1,
                       inv.logit(alpha[y[i]] + eta[i])) -
                ifelse(y[i] == 1, 0,
                       inv.logit(alpha[y[i] - 1] + eta[i]))
  }

  # compute log-likelihood: sum of log(p)
  return(sum(log(probs)))
}

