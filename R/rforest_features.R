

#' Feature Selection by Ordinal Random Forest Importance
#' 
#' Uses the OrdinalForest and Boruta packages to select important features from a dataframe
#' 
#' @param y_name: character, the factor column in data containing the ordinal y values
#' @param data: a dataframe containing the outcome variable and the features to be selected
#'
#' @return A list containing: \describe{
#'      \item{y.unique}{A vector of the trained scores from the ordinal random forest model}
#'      \item{forest}{The fitted ordinal random forest model}
#'      \item{boruta}{The result of the Boruta feature selection}
#'      \item{important.features}{A vector of the names of the important features}
#'      \item{times}{A list of the computation times for the two major steps}
#' }
#'
#' @export
#' 
#' @examples
#'
#' dat <- simulate.data(
#'   n = 200,
#'   alpha = seq(0.5, 4, length=3),
#'   beta = c(rep(1,5), rep(0,10)))
#'
#' result <- randomforest.features("y", dat)
#' result$important.features
#'
randomforest.features <- function(y_name, data, pValue=0.01, maxRuns=100, doTrace=0, holdHistory=FALSE, ...) {

  start <- Sys.time()
  forest <- ordinalForest::ordfor(y_name, data, ...)
  time.ordinalForest <- Sys.time() - start

  y <- data[[y_name]]
  data[y_name] <- NULL
  
  borders <- forest$bordersbest
  borders.avg <- (borders[-1] + borders[-length(borders)])/2

  ymetric <- qnorm(borders.avg)[as.numeric(y)]
  
  start <- Sys.time()
  boruta <- Boruta::Boruta(data, ymetric, pValue=pValue, maxRuns=maxRuns, doTrace=doTrace, holdHistory=holdHistory)
  time.Boruta <- Sys.time() - start

  important.features <- names(boruta$finalDecision[boruta$finalDecision == "Confirmed"])

  list(y.unique = qnorm(borders.avg), 
       forest = forest, 
       boruta = boruta, 
       important.features = important.features,
       times = list(ordinalForest = time.ordinalForest,
                    Boruta = time.Boruta))
}

