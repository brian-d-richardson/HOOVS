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
#' @export
prox.proj <- function(z, lambda, m, ind = NULL) {

  if (is.null(ind)) {ind <- 1:length(z)}

  diff <- abs(z[ind]) - m * lambda

  c(z[-ind], sign(z[ind]) *
      ifelse(diff < 0, 0, diff))
}
