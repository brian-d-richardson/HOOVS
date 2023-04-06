#' Inverse Logit Derivative
#'
#' This function computes derivative of inverse logit at all elements in a numeric vector
#'
#' @param x: a numeric vector
#'
#' @return the inverse logit derivative
#'
#' @export
d.inv.logit <- function(x) {
  exp(x) / (1 + exp(x)) ^ 2
}
