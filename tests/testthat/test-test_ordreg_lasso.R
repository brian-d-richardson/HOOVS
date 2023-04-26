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

test_that("non-zero covariates dimension correct", {
  expect_equal(length(res.ordreg$n.nonzero), length(lambdas))
})

test_that("log-likelihood dimension correct", {
  expect_equal(length(res.ordreg$loglik.val), length(lambdas))
})

test_that("kappa dimension correct", {
  expect_equal(length(res.ordreg$kappa), length(lambdas))
})

test_that("BIC dimension correct", {
  expect_equal(length(res.ordreg$bic), length(lambdas))
})

test_that("predictions dimension correct", {
  expect_equal(dim(res.ordreg$predictions), c(length(lambdas), n))
})




