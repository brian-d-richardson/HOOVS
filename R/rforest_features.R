
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
       forest = fit.forest, 
       boruta = boruta, 
       important.features = important.features,
       times = list(ordinalForest = time.ordinalForest,
                    Boruta = time.Boruta))
}
