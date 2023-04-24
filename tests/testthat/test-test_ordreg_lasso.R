library(psych)
# Simulate Data
set.seed(1)
n <- 1000
p <- 50
J <- 3
lambdas <- seq(0, 0.5, 0.05)
alpha <- seq(.5, 4, length = J - 1)
beta <- rep(0, p)
beta[1: floor(p / 2)] <- 1

dat <- simulate.data(
  n = 1000,
  alpha = alpha,
  beta = beta)

# Run ordinal regression
res.ordreg <- ordreg.lasso(
  formula = y ~ .,
  data = dat,
  lambdas
)

# Create package tests
test_that("alpha dimension correct", {
  expect_equal(dim(res.ordreg$alpha), c(length(lambdas), J-1))
})

test_that("beta dimension correct", {
  expect_equal(dim(res.ordreg$beta), c(length(lambdas), p))
})


