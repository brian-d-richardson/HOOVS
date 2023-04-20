
#' Inverse Logit Derivative
#'
#' This function computes derivative of inverse logit at all elements in a numeric vector
#'
#' @param x: a numeric vector
#'
#' @return the inverse logit derivative
#'
#' @export
inv.logit.d <- function(x) {
  exp(x) / (1 + exp(x)) ^ 2
}

#' Inverse logit function
#'
#' This function computes the inverse logit of all elements in a numeric vector
#'
#' @param x: a numeric vector
#'
#' @return the inverse logit values
#'
#' @export
inv.logit <- function(x) {
  exp(x) / (1 + exp(x))
}

#' compute zeta from alpha
#'
#' This function computes the parameter vector zeta from the vector alpha
#'
#' @param zeta: a numeric vector
#'
#' @return alpha vector
#'
#' @keywords internal
ordreg.alphatozeta <- function(alpha) {
  c(alpha[1], log(diff(alpha)))
}


#' compute alpha from zeta
#'
#' This function computes the parameter vector alpha from the vector zeta
#'
#' @param zeta: a numeric vector
#'
#' @return alpha vector
#'
#' @keywords internal
ordreg.zetatoalpha <- function(zeta) {
  cumsum(c(zeta[1], exp(zeta[-1])))
}

#' compute derivative of alpha w.r.t. zeta
#'
#' This function computes the derivative of alpha w.r.t. zeta
#'
#' @param zeta: a numeric vector of length J-1
#'
#' @return a derivative matrix of size (J-1) x (J-1)
#'
#' @keywords internal
ordreg.zetatoalpha.d <- function(zeta) {
  d <- matrix(data = rep(c(1, exp(zeta[-1])), each = J - 1),
                   nrow = J - 1)
  d[upper.tri(d)] <- 0
  return(d)
}

#' Ordinal Logistic Regression Log-likelihood Function
#'
#' This function computes the log-likelihood value from the ordinal regression model (with a logit link) given data and parameters
#'
#' @param y: outcome vector; a numeric vector of length n with values in 1, ..., J corresponding to the J ordinal outcome categories
#' @param x: covariate matrix; n x p matrix with numeric elements
#' @param alpha: category specific intercepts; numeric vector of length J-1; defaults to NULL, in which case zeta must be supplied instead
#' @param zeta: used to generate category specific intercepts alpha = sum(exp(zeta)); numeric vector of length J-1
#' @param beta: slope parameters; numeric vector of length p
#'
#' @return the log-likelihood value
#'
#' @export
ordreg.loglik <- function(y, x, zeta, beta, alpha = NULL) {

  # number of observations
  n <- length(y)

  # number of ordered factor levels
  J <- max(y)

  # category-specific intercepts
  if (is.null(alpha)) { alpha <- ordreg.zetatoalpha(zeta) }

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
ordreg.loglik.d <- function(y, x, zeta, beta, alpha = NULL) {
  
  # number of covariates
  p <- ncol(x)

  # number of observations
  n <- length(y)

  if (is.null(alpha)) {

    # category-specific intercepts
    alpha <- ordreg.zetatoalpha(zeta)

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

      psi_ij_1 <- inv.logit.d(alpha[y[i] - 1] + eta[i])

      g.alpha[i, y[i] - 1] <- - psi_ij_1 / p_ij

      g.beta[i, ] <- (0 - psi_ij_1) * x[i,] / p_ij

    } else if (y[i] == 1) {

      p_ij <- inv.logit(alpha[y[i]] + eta[i])

      psi_ij <- inv.logit.d(alpha[y[i]] + eta[i])

      g.alpha[i, y[i]] <- psi_ij / p_ij

      g.beta[i, ] <- (psi_ij - 0) * x[i,] / p_ij

    } else {

      p_ij <- inv.logit(alpha[y[i]] + eta[i]) -
              inv.logit(alpha[y[i] - 1] + eta[i])

      psi_ij <- inv.logit.d(alpha[y[i]] + eta[i])
      psi_ij_1 <- inv.logit.d(alpha[y[i] - 1] + eta[i])

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
    return(c(apply(g.alpha, 2, sum) %*% ordreg.zetatoalpha.d(zeta),
             apply(g.beta, 2, sum)))
  }
}

#' Proximal Projection Operator
#'
#' This function computes the proximal projection operator for the LASSO penalty used in the proximal gradient descent algorithm
#'
#' @param z: a numeric vector to be projected
#' @param lambda: LASSO penalty parameter; a non-negative number
#' @param m: step size; a non-negative number
#' @param ind: an optional vector with indices indicating which components of z should be projected; defaults to NULL, in which case all components of z are projected
#'
#' @return a numeric vector of the same length as z
#'
#' @keywords internal
prox.proj <- function(z, lambda, m, ind = NULL) {

  if (is.null(ind)) {ind <- 1:length(z)}

  diff <- abs(z[ind]) - m * lambda

  c(z[-ind], sign(z[ind]) *
      ifelse(diff < 0, 0, diff))
}


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
#' @keywords internal
ordreg.prox.grad.desc <- function(y, x, zeta0, beta0, lambda,
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
  ll.k <- ordreg.loglik(y = y, x = x, zeta = theta.k[zet], beta = theta.k[bet]) / -n

  # (-1/n) gradient of log-likelihood at current estimate
  g.k <- ordreg.loglik.d(y = y, x = x, zeta = theta.k[zet], beta = theta.k[bet]) / -n

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
      ll <- ordreg.loglik(y = y, x = x, zeta = theta[zet], beta = theta[bet]) / -n

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
        g.k <- ordreg.loglik.d(y = y, x = x, zeta = theta[zet], beta = theta[bet]) / -n

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


#' Ordinal Logistic Regression Prediction
#'
#' This function computes predicted values given ordinal regression coefficients and a data frame with covariates.
#'
#' @param alpha: category specific intercepts; numeric vector of length J-1
#' @param beta: slope parameters; numeric vector of length p
#' @param x: covariate matrix; n x p matrix with numeric elements
#'
#' @return a numeric vector of length n with predicted y values in {1, ..., J}
#' 
#' @export

ordreg.predict <- function(alpha, beta, x) {
  
  # sample size
  n <- nrow(x)
  
  # number of outcome categories
  J <- length(alpha) + 1
  
  # compute cumulative probabilities P(y_i <= j)
  cum.probs <- inv.logit(matrix(rep(x %*% beta, each = J-1),
                                ncol = J - 1, byrow = T) +
                           matrix(rep(alpha, each = n),
                                  nrow = n, byrow = F))
  
  # compute cell probabilities P(y_i == j)
  probs <- apply(X = cbind(0, cum.probs, 1),
                 MARGIN = 1,
                 FUN = diff)
  
  # predict values based on maximum cell probability
  apply(X = probs,
        MARGIN = 2,
        FUN = which.max)
}


#' Ordinal Logistic Regression with LASSO Penalty
#'
#' This function fits LASSO-penalized ordinal regression models over a grid of penalty parameters.
#'
#' @param formula: a symbolic description of the model to be fitted; an object of class `"formula"`
#' @param data: a data frame containing the variables in the model
#' @param lambdas: vector of LASSO penalty parameters; a vector of non-negative numbers; default is 0, which corresponds to no penalization
#' @param return.cov: an indicator for whether asymptotic covariance matrix of parameter estimates should be output; defaults to FALSE
#'
#' @return a list with the following elements:
#' \itemize{
#' \item{lambdas: vector of LASSO penalty parameters (sorted in descending order)}
#' \item{alpha: matrix of alpha estimates with each row corresponding to a lambda value}
#' \item{beta: matrix of beta estimates with each row corresponding to a lambda value}
#' \itme{n.nonzero: number of nonzero parameters for each lambda value; a vector of non-negative integers}
#' \item{loglik.val: log-likelihood values at convergence (not including the LASSO penalty) for each lambda value; a numeric vector}
#' \item{bic: Bayesian information critetion value at convergence for each lambda value; a numeric vector}
#' \item{predictions: matrix of predicted y values with each row corresponding to a lambda value}
#' \item{kappa: vector of weighted kappa values for each lambda value}
#' \item{cov: asymptotic covariance matrix of parameter estimates from unpenalized fit (lambda = 0)}
#' }
#' 
#' @importFrom numDeriv jacobian
#' @importFrom psych cohen.kappa
#'
#' @export
ordreg.lasso <- function(formula, data, lambdas = 0, return.cov = FALSE) {
  
  # if return.cov = TRUE, verify that lambdas includes 0
  # (the covariance matrix for parameter estimates is specific to lambda = 0)
  if (return.cov) {
    if (!(0 %in% lambdas)) {
      stop("asymptotic covariance is valid only for lambda = 0")
    } else if (length(lambdas) > 1) {
      warning("asymptotic covariance is not valid if parameter tuning is done on same data")
    }
  }
  
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
  lambdas <- sort(lambdas, decreasing = T)

  # initialize outputs
  alpha <- matrix(data = NA, nrow = L, ncol = J-1,
                  dimnames = list(paste0("lambda=", lambdas),
                                  paste0("alpha", 1:(J-1))))
  
  beta <- matrix(data = NA, nrow = L, ncol = p,
                 dimnames = list(paste0("lambda=", lambdas),
                                 colnames(x)))
  
  predictions <- matrix(data = NA, nrow = L, ncol = n,
                        dimnames = list(paste0("lambda=", lambdas),
                                        paste0("y", 1:n)))
  
  n.nonzero <- loglik.val <- bic <- kappa <- numeric(L)
  names(n.nonzero) <- names(loglik.val) <- names(bic) <- names(kappa) <- paste0("lambda=", lambdas) 
  
  cov <- NULL

  # initialize starting values
  zeta0 <- ordreg.alphatozeta(seq(.1, 1, length = J - 1))
  beta0 <- rep(0, p)

  # loop through lambdas
  for (l in 1:L) {

    # fit model using PGD algorithm
    pgd.fit <- ordreg.prox.grad.desc(
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
    alpha[l, ] <- ordreg.zetatoalpha(pgd.fit$zeta)
    beta[l, ] <- pgd.fit$beta
    loglik.val[l] <- pgd.fit$loglik.val
    bic[l] <- -2 * pgd.fit$loglik.val +
            log(n) * n.nonzero[l]
    
    # compute predicted values
    predictions[l,] <- ordreg.predict(alpha = alpha[l,],
                                      beta = beta[l,],
                                      x = x)
    
    # compute weighted kappa
    kappa[l] <- psych::cohen.kappa(x = matrix(c(y, predictions[l,]),
                                              ncol = 2, byrow = F))$weighted.kappa
  }
  
  if (return.cov) {
    
    # observed information matrix
    obs.inf <- numDeriv::jacobian(
      func = function(theta) {
        ordreg.loglik.d(y = y, x = x,
                        alpha = theta[1:(J-1)],
                        beta = theta[-(1:(J-1))])
      }, x = c(alpha[which(lambdas == 0), ],
               beta[which(lambdas == 0), ])
    )
    
    # covariance matrix
    cov <- solve(-obs.inf)
    colnames(cov) <- rownames(cov) <- c(paste0("alpha", 1:(J-1)),
                                        colnames(x))
  }
  
  colnames(beta)
  
  return(list("lambdas" = lambdas,
              "alpha" = alpha,
              "beta" = beta,
              "n.nonzero" = n.nonzero,
              "loglik.val" = loglik.val,
              "bic" = bic,
              "predictions" = predictions,
              "kappa" = kappa,
              "cov" = cov))
}

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
ordreg.slow.fit <- function(y, x, zeta0, beta0, lambda) {

  n <- length(y)

  # maximize likelihood
  opt.res <- optim(
    par = c(zeta0, beta0),
    fn = function(theta) {
      beta <- theta[-(1:(J - 1))]
      zeta <- theta[1:(J - 1)]
      -(1 / n) * (
        # log-likelihood
        ordreg.loglik(zeta = zeta,
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

