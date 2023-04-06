#' Ordinal Logistic Regression with LASSO Penalty
#'
#' This function fits LASS)-penalized ordinal regression models over a grid of penalty parameters.
#'
#' @param formula: a symbolic description of the model to be fitted; an object of class `"formula"`
#' @param data: a data frame containing the variables in the model
#' @param lambdas: vector of LASSO penalty parameters; a vector of non-negative numbers; default is 0, which corresponds to no penalization
#'
#' @return a list with the following elements:
#' \itemize{
#' \item{lambda: vector of LASSO penalty parameters}
#' \item{alpha: matrix of alpha estimates with each row corresponding to a lambda value}
#' \item{beta: matrix of beta estimates with each row corresponding to a lambda value}
#' \itme{n.nonzero: number of nonzero parameters for each lambda value; a vector of non-negative integers}
#' \item{loglik.val: log-likelihood values at convergence (not including the LASSO penalty) for each lambda value; a numeric vector}
#' \item{bic: Bayesian information critetion value at convergence for each lambda value; a numeric vector}
#' }
#'
#' @export
ordreg.lasso <- function(formula, data, lambdas = 0) {

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

  # number of covariates
  p <- ncol(x)

  # number of lambda values
  L <- length(lambdas)

  # sort lambdas in ascending order
  lambdas <- sort(lambdas, decreasing = F)

  # initialize outputs
  alpha <- matrix(data = NA, nrow = L, ncol = J-1)
  beta <- matrix(data = NA, nrow = L, ncol = p)
  n.nonzero <- numeric(L)
  loglik.val <- numeric(L)
  bic <- numeric(L)

  # initialize starting values
  zeta0 <- get.zeta(seq(.1, 1, length = J - 1))
  beta0 <- rep(0, p)

  # loop through lambdas
  for (l in 1:L) {

    # fit model using PGD algorithm
    pgd.fit <- prox.grad.desc(
      y = y,
      x = x,
      zeta0 = zeta0,
      beta0 = beta0,
      lambda = lambdas[l]
    )

    # update initial values for next iteration
    zeta0 <- pgd.fit$zeta
    beta0 <- pgd.fit$beta

    # number of nonzero parameters
    n.nonzero[l] <- (sum(pgd.fit$beta != 0) + J - 1)

    # add results for current lambda to output
    alpha[l, ] <- get.alpha(pgd.fit$zeta)
    beta[l, ] <- pgd.fit$beta
    loglik.val[l] <- pgd.fit$loglik.val
    bic[l] <- -2 * pgd.fit$loglik.val +
            log(n) * n.nonzero[l]

  }

  return(list("alpha" = alpha,
              "beta" = beta,
              "n.nonzero" = n.nonzero,
              "loglik.val" = loglik.val,
              "bic" = bic))
}
