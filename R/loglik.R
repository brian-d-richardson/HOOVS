#' Ordinal Logistic Regression Log-likelihood Function
#'
#' This function computes the log-likelihood value from the ordinal regression model (with a logit link) given data and parameters
#'
#' @param y: outcome vector; an ordered factor vector of length n with J levels
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
  J <- nlevels(y)

  # category-specific intercepts
  alpha = cumsum(exp(zeta))

  # convert factor levels to numeric values
  Y <- numeric(n)
  for (j in 1:J) {
    Y[y == levels(y)[j]] <- j
  }

  # linear predictor
  eta <- x %*% beta

  # for each observation, compute P(Z <= j | xi) - P(Z <= j - 1 | xi)
  probs <- numeric(length(Y))
  for (i in 1:length(Y)) {
    probs[i] <- ifelse(Y[i] == J, 1,
                       inv.logit(alpha[Y[i]] + eta[i])) -
                ifelse(Y[i] == 1, 0,
                       inv.logit(alpha[Y[i] - 1] + eta[i]))
  }

  # compute log-likelihood: sum of log(p)
  return(sum(log(probs)))
}

