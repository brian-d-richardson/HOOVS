#' Ordinal Logistic Regression Log-likelihood Gradient
#'
#' This function computes the gradient of the log-likelihood value from the ordinal regression model (with a logit link) given data and parameters (zeta and beta)
#'
#' @param y: outcome vector; a numeric vector of length n with values in 1, ..., J corresponding to the J ordinal outcome categories
#' @param x: covariate matrix; n x p matrix with numeric elements
#' #' @param alpha: category specific intercepts; numeric vector of length J-1; defaults to NULL, in which case zeta must be supplied instead
#' @param zeta: used to generate category specific intercepts alpha = sum(exp(zeta)); numeric vector of length J-1
#' @param beta: slope parameters; numeric vector of length p
#'
#' @return the log-likelihood gradient value with respect to c(zeta, beta) or c(alpha, beta) depending on whether zeta or alpha is supplied.
#'
#' @export
gloglik <- function(y, x, zeta, beta, alpha = NULL) {

  # number of observations
  n <- length(y)

  if (is.null(alpha)) {

    # category-specific intercepts
    alpha <- get.alpha(zeta)

    # indicator for whether gradient should be taken w.r.t alpha
    wrt.alpha <- FALSE

  } else {
    wrt.alpha <- TRUE
  }

  # number of ordered factor levels
  J <- length(alpha) + 1

  # linear predictor: a vector of length n
  eta <- x %*% beta

  # for each observation y compute:
    # contribution to grad w.r.t. alpha
    # contribution to grad w.r.t. beta
  g.alpha <- matrix(data = 0, nrow = n, ncol = J - 1)
  g.beta <- matrix(data = 0, nrow = n, ncol = p)
  for (i in 1:n) {

    if (y[i] == J) {

      p_ij <- 1 - inv.logit(alpha[y[i] - 1] + eta[i])

      psi_ij_1 <- d.inv.logit(alpha[y[i] - 1] + eta[i])

      g.alpha[i, y[i] - 1] <- - psi_ij_1 / p_ij

      g.beta[i, ] <- (0 - psi_ij_1) * x[i,] / p_ij

    } else if (y[i] == 1) {

      p_ij <- inv.logit(alpha[y[i]] + eta[i])

      psi_ij <- d.inv.logit(alpha[y[i]] + eta[i])

      g.alpha[i, y[i]] <- psi_ij / p_ij

      g.beta[i, ] <- (psi_ij - 0) * x[i,] / p_ij

    } else {

      p_ij <- inv.logit(alpha[y[i]] + eta[i]) -
              inv.logit(alpha[y[i] - 1] + eta[i])

      psi_ij <- d.inv.logit(alpha[y[i]] + eta[i])
      psi_ij_1 <- d.inv.logit(alpha[y[i] - 1] + eta[i])

      g.alpha[i, y[i]] <- psi_ij / p_ij
      g.alpha[i, y[i] - 1] <- - psi_ij_1 / p_ij

      g.beta[i, ] <- (psi_ij - psi_ij_1) * x[i,] / p_ij

    }
  }

  # return gradient of log-likelihood
  if (wrt.alpha) {
    return(c(apply(g.alpha, 2, sum),
             apply(g.beta, 2, sum)))
  } else {
    return(c(apply(g.alpha, 2, sum) %*% d.alpha(zeta),
             apply(g.beta, 2, sum)))
  }
}

