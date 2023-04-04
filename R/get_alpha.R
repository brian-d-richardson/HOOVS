#' compute alpha from zeta
#'
#' This function computes the parameter vector alpha from the vector zeta
#'
#' @param zeta: a numeric vector
#'
#' @return alpha vector
#'
#' @export
get.alpha <- function(zeta) {
  cumsum(exp(zeta))
}
