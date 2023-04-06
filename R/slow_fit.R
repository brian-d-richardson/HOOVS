#' Slow Optimization Procedure
#'
#' Slow procedure to fit LASSO-penalized ordinal regression model; only included for purposes of comparing to faster approaches (like PGD).
#'
#' @param y: outcome vector; a numeric vector of length n with values in 1, ..., J corresponding to the J ordinal outcome categories
#'  @param x: covariate matrix; n x p matrix with numeric elements
#'  @param zeta0: starting value for zeta; numeric vector of length J-1
#'  @param beta0: starting value for beta; numeric vector of length p
#'  @param lambda: LASSO penalty parameter; a non-negative number
#'
#'#' @return a list with the following elements:
#' \itemize{
#' \itme{zeta: the estimated zeta parameters; a numeric vector of length J-1}
#' \item{beta: the estimated slopes; a numeric vector of length p}
#' \item{loglik.val: the log-likelihood value at convergence}
#' \item{obj.val: the objective function value at convergence}
#' \item{n.iterations: the number of outer loop iterations in the PGD algorithm}
#' }
#'
#' @export
slow.fit <- function(y, x, zeta0, beta0, lambda) {

  n <- length(y)

  # maximize likelihood
  opt.res <- optim(
    par = c(zeta0, beta0),
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

  zeta = opt.res$par[1:(J-1)]
  beta = opt.res$par[-(1:(J-1))]

  return(list("zeta" = zeta,
              "beta" = beta,
              "loglik.val" = -n * (opt.res$value - lambda * sum(abs(beta))),
              "obj.val" = opt.res$value,
              "n.iterations" = unname(opt.res$counts[1])))
}
