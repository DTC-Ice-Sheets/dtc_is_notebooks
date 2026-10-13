library(hdftsa)

model.coeff <- function(data, delta=24, p0=2, L=3, h=3){
  # it fits multivariate functional time series and forecasts
  # args:
  #   data: matrix of coefficients with dim nbasis x time-points
  #   delta: num of time points to be considered as a single curve
  #   p0: number of principal component to retain
  #   L: The number of factors to retain in the second step dimension reduction
  #   h: Forecast horizon
  # values: a list
  #
  t0 <- Sys.time()

  tt <- dim(data)[2] # num of time points
  nn <- tt/delta # num of curves
  nbasis <- dim(data)[1] # num of basis functions
  Coeff <- data
  aux.training <- matrix(NA, delta, nn)
  Mult.FTS <- rep(list(aux.training),nbasis)

  for (k in 1:nbasis) {
    Mult.FTS[[k]] <- matrix(Coeff[k, ], delta, nn)
  }

  #cat("\n\n",
  #    'Fitting functional time series model.\n',
  #    'Time remaining: a few seconds more.\n')

  mts_model = hdfpca(Mult.FTS, order = p0, r = L)
  # forecast of the mult time series
  Coeff_forecast = forecast.hdfpca(object = mts_model, h = h)
  t1 <- Sys.time()
  return(list(y=data,
              model=mts_model,
              forecast=Coeff_forecast,
              cpu=t1-t0))
}


model.mu <- function(data, delta=24, rangeval=c(0,1), p0=4, h=3){
  # arg:
  #   data: a vector with lenght= time-points
  #   delta: num of time points to be considered as a single curve
  #   p0: number of principal component to retain
  #   h: Forecast horizon
  tt <- length(data)
  nn <- tt/delta
  mu <- matrix(data, delta, nn)

  s=seq(rangeval[1],rangeval[2],length.out = delta)
  colnames(mu) <- 1:nn
  fts_mu <- fts(s,mu)
  # fit a principal component model to the funct time series (mu)
  modelfts_mu <- ftsm(fts_mu, order=p0, mean=TRUE, method = "classical")
  forecast_mu <- forecast(modelfts_mu, h=h, method="arima" )

  return(list(y=data,
              model=modelfts_mu,
              forecast=forecast_mu))
}
