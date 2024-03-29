% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ordreg_lasso.R
\name{ordreg.prox.grad.desc}
\alias{ordreg.prox.grad.desc}
\title{Proximal Gradient Descent Algorithm}
\usage{
ordreg.prox.grad.desc(
  y,
  x,
  zeta0,
  beta0,
  lambda,
  m = 5,
  a = 0.8,
  eps = 1e-08,
  maxit = 1000,
  print.updates = FALSE
)
}
\arguments{
\item{y:}{outcome vector; a numeric vector of length n with values in 1, ..., J corresponding to the J ordinal outcome categories}

\item{x:}{covariate matrix; n x p matrix with numeric elements}

\item{zeta0:}{starting value for zeta; numeric vector of length J-1}

\item{beta0:}{starting value for beta; numeric vector of length p}

\item{lambda:}{LASSO penalty parameter; a non-negative number}

\item{m:}{initial step size; a positive number; defaults to 5}

\item{a:}{step-size decrement factor; a number in the interval (0, 1); defaults to 0.8}

\item{eps:}{a stopping threshold for the relative change in successive evaluations of objective function; a positive number; defaults to 1E-8}

\item{maxit:}{maximum number of iterations}

\item{print.updates:}{an indicator for whether iteration updates should be output; defaults to FALSE}
}
\value{
a list with the following elements:
\itemize{
\itme{zeta: the estimated zeta parameters; a numeric vector of length J-1}
\item{beta: the estimated slopes; a numeric vector of length p}
\item{loglik.val: the log-likelihood value at convergence}
\item{obj.val: the objective function value at convergence}
\item{n.iterations: the number of outer loop iterations in the PGD algorithm}
}
}
\description{
Proximal gradient descent algorithm for
}
\keyword{internal}
