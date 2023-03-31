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
