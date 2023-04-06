#' compute derivative of alpha w.r.t. zeta
#'
#' This function computes the derivative of alpha w.r.t. zeta
#'
#' @param zeta: a numeric vector of length J-1
#'
#' @return a derivative matrix of size (J-1) x (J-1)
#'
#' @export
d.alpha <- function(zeta) {
  d <- matrix(data = rep(c(1, exp(zeta[-1])), each = J - 1),
                   nrow = J - 1)
  d[upper.tri(d)] <- 0
  return(d)
}
