#' Proximal Gradient Descent Algorithm
#'
#' Proximal gradient descent algorithm for
#'
#' @param y: outcome vector; a numeric vector of length n with values in 1, ..., J corresponding to the J ordinal outcome categories
#' @param x: covariate matrix; n x p matrix with numeric elements
#' @param zeta0: starting value for zeta; numeric vector of length J-1
#' @param beta0: starting value for beta; numeric vector of length p
#' @param lambda: LASSO penalty parameter; a non-negative number
#' @param m: initial step size; a positive number; defaults to 5
#' @param a: step-size decrement factor; a number in the interval (0, 1); defaults to 0.8
#' @param eps: a stopping threshold for the relative change in successive evaluations of objective function; a positive number; defaults to 1E-8
#' @param maxit: maximum number of iterations
#' @param print.updates: an indicator for whether iteration updates should be output; defaults to FALSE
#'
#' @return a list with the following elements:
#' \itemize{
#' \itme{zeta: the estimated zeta parameters; a numeric vector of length J-1}
#' \item{beta: the estimated slopes; a numeric vector of length p}
#' \item{loglik.val: the log-likelihood value at convergence}
#' \item{obj.val: the objective function value at convergence}
#' \item{n.iterations: the number of outer loop iterations in the PGD algorithm}
#' }
#'
#' @export
prox.grad.desc <- function(y, x, zeta0, beta0, lambda,
                           m = 5, a = 0.8, eps = 1E-8,
                           maxit = 1000, print.updates = FALSE) {

  # sample size
  n <- length(y)

  # initiate difference in successive evaluations
  delta <- 10^3

  # indices for zeta and beta
  zet <- 1:(length(zeta0))
  bet <- length(zeta0) + 1:length(beta0)

  # initiate parameters
  theta.k <- c(zeta0, beta0)

  # (-1/n) log-likelihood at current estimate
  ll.k <- loglik(y = y, x = x, zeta = theta.k[zet], beta = theta.k[bet]) / -n

  # (-1/n) gradient of log-likelihood at current estimate
  g.k <- gloglik(y = y, x = x, zeta = theta.k[zet], beta = theta.k[bet]) / -n

  # objective function at current estimate
  obj.k <- ll.k + lambda * sum(abs(theta.k[bet]))

  # counter for iterations in outer loop
  outer.its <- 0

  ### START OUTER LOOP ###
  while (outer.its <= maxit & delta >= eps) {

    # print updates
    if (print.updates) {
      print(paste0("outer iteration: ", outer.its,
                   "; current objective function value: ", obj.k,
                   "; current step size: ", m))
    }

    # update outer iteration counter
    (outer.its <- outer.its + 1)

    # counter for iterations in inner loop
    inner.its <- 0

    ### START INNER LOOP FOR Kth ITERATION ###
    while (inner.its < maxit) {

      # update inner iteration counter
      inner.its <- inner.its + 1

      # update parameter estimate
      theta <- prox.proj(z = theta.k - m * g.k,
                         m = m,
                         lambda = lambda,
                         ind = bet)

      # (-1/n) log-likelihood at proposed new theta
      ll <- loglik(y = y, x = x, zeta = theta[zet], beta = theta[bet]) / -n

      # objective function at proposed new theta
      obj <- ll + lambda * sum(abs(theta[bet]))

      # compute local quadratic approximation
      lqa.k <- ll.k + g.k %*% (theta - theta.k) +
        (0.5 / m) * (theta - theta.k) %*% (theta - theta.k)

      # check for majorization
      if (ll <= lqa.k) {

        # update difference in successive evaluations of objective function
        delta <- abs((obj.k - obj) / obj.k)

        # update theta.k, ll.k, obj.k, and g.k
        theta.k <- theta
        ll.k <- ll
        obj.k <- obj
        g.k <- gloglik(y = y, x = x, zeta = theta[zet], beta = theta[bet]) / -n

        break
      } else {
        # shrink step size and repeat if majorization did not occur
        m <- a * m
      }
    } ### END INNER LOOP FOR Kth ITERATION ###
  } ### END OUTER LOOP ###

  if (outer.its >= maxit) {
    warning("maximum number of iterations reached without convergence")
  }

  return(list("zeta" = theta[zet],
              "beta" = theta[bet],
              "loglik.val" = - n * ll,
              "obj.val" = obj,
              "n.iterations" = outer.its))
}
