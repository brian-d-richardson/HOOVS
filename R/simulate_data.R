#' Simulate Data from Ordinal Logistic Regression Model
#'
#' This function simulates a data set from an ordinal regression model
#'
#' @param n: number of observations to be simulated: a positive integer
#' @param alpha: category specific intercepts for the lowest J-1 outcome categories; a numeric vector of length J-1
#' @param beta: slope coefficients; a numeric vector of length p
#'
#' @return a data frame with n rows and the following p + 1 columns
#'
#' @export
simulate.data <- function(n, alpha, beta) {

  # number of outcome categories
  J <- length(alpha) + 1

  # number of covariates
  p <- length(beta)

  # create n x p covariate matrix
  X <- matrix(nrow = n,
              ncol = p,
              data = rnorm(n*p))

  # linear predictors
  eta0 <- X %*% beta

  # compute the cumulative probabilities for each of J categories and n subjects
  cum.probs <- matrix(nrow = n,
                      ncol = J)
  for (j in 1:(J-1)) {
    cum.probs[, j] <- inv.logit(alpha[j] + eta0)
  }
  cum.probs[, J] <- 1

  # compute the probabilities for each of J categories and n subjects
  probs <- cum.probs - cbind(0, cum.probs[, -J])

  # simulate outcomes
  y <- numeric(n)
  for (i in 1:n) {
    y[i] <- sample(x = 1:J,
                   size = 1,
                   prob = probs[i,])
  }

  # combine into one data set
  dat <- data.frame(cbind(y, X))
  colnames(dat) <- c("y", paste0("X", 1:p))
  dat$y <- factor(dat$y, levels = 1:J, ordered = T)

  return(dat)
}
