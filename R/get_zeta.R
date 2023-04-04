#' compute zeta from alpha
#'
#' This function computes the parameter vector zeta from the vector alpha
#'
#' @param zeta: a numeric vector
#'
#' @return alpha vector
#'
#' @export
get.zeta <- function(alpha) {
  c(alpha[1], log(diff(alpha)))
}
