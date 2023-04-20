HOOVS: High Dimensional Ordinal Outcome Variable Selection
================
Ben Bodek, Brian Chen, Forrest Hurley, Brian Richardson, Emmanuel
Rockwell

## Description

The `HOOVS` package (“High-Dimensional Ordinal Outcome Variable
Selection”) is for a group project for Bios 735 (statistical computing).
The goal of the project is to develop two methods to perform variable
selection on a high-dimensional data set with an ordinal outcome. The
first method is a LASSO-penalized ordinal regression model and the
second is a random forest model.

## Installation

Installation of the `HOOVS` from GitHub requires the
[`devtools`](https://www.r-project.org/nosvn/pandoc/devtools.html)
package and can be done in the following way.

``` r
# Install the package
devtools::install_github("brian-d-richardson/HOOVS")
```

``` r
# Then load it
library(HOOVS)
```

Other packages used in this README can be loaded in the following chunk.

``` r
suppressPackageStartupMessages(if (!require(dplyr)) {install.packages("dplyr")})
suppressPackageStartupMessages(if (!require(tidyr)) {install.packages("tidyr")})
suppressPackageStartupMessages(if (!require(ggplot2)) {install.packages("ggplot2")})
suppressPackageStartupMessages(if (!require(ordinalNet)) {install.packages("ordinalNet")})
suppressPackageStartupMessages(if (!require(foreign)) {install.packages("foreign")})
suppressPackageStartupMessages(if (!require(devtools)) {install.packages("devtools")})
suppressPackageStartupMessages(if (!require(tictoc)) {install.packages("tictoc")})
suppressPackageStartupMessages(if (!require(psych)) {install.packages("psych")})

load_all()
#> ℹ Loading HOOVS
#For reproducibility
set.seed(1)
```

# Method 1: LASSO-Penalized Ordinal Regression Model

We begin with a theoretical introduction of the ordinal regression model
and how the `HOOVS` package calculates parameter estimates.

## Ordinal Regression Model Setup

Suppose that, for observation *i* = 1, …, *n*, the ordinal outcome
*Y*<sub>*i*</sub> given the covariate vector **x**<sub>*i*</sub> has a
multinomial distribution with *J* outcome categories and probabilities
of success
*π*<sub>1</sub>(**x**<sub>*i*</sub>), …, *π*<sub>*J*</sub>(**x**<sub>*i*</sub>).
That is,

*Y*<sub>*i*</sub>\|**x**<sub>*i*</sub> ∼ multinomial{1; *π*<sub>1</sub>(**x**<sub>*i*</sub>), …, *π*<sub>*J*</sub>(**x**<sub>*i*</sub>)}.

The cumulative probability for subject *i* and ordinal outcome category
*j* is
$P(Y\_i \\leq j \| \\pmb{x}\_i) = \\sum\_{k=1}^j \\pi\_k(\\pmb{x}\_i)$.
Note that by definition
*P*(*Y*<sub>*i*</sub> ≤ *J*\|**x**<sub>*i*</sub>) = 1.

The following proportional odds model relates the cumulative probability
for subject *i* and ordinal outcome category *j* to the covariates
**x**<sub>*i*</sub> via the parameters
**α** = (*α*<sub>1</sub>, …, *α*<sub>*J* − 1</sub>)<sup>*T*</sup> and
**β** = (*β*<sub>1</sub>, …, *β*<sub>*p*</sub>)<sup>*T*</sup> with a
logit link function.

logit{*P*(*Y*<sub>*i*</sub> ≤ *j*\|**x**<sub>*i*</sub>)} = *α*<sub>*j*</sub> + **x**<sub>*i*</sub><sup>*T*</sup>**β**.

In this model, *α*<sub>1</sub>, …, *α*<sub>*J* − 1</sub> are outcome
category-specific intercepts for the first *J* − 1 ordinal outcome
categories and *β*<sub>1</sub>, …, *β*<sub>*p*</sub> are the slopes
corresponding to the *p* covariates. Since the cumulative probabilities
must be increasing in *j*, i.e.,
*P*(*Y*<sub>*i*</sub> ≤ *j*\|**x**<sub>*i*</sub>) &lt; *P*(*Y*<sub>*i*</sub> ≤ *j* + 1\|**x**<sub>*i*</sub>),
we require that *α*<sub>1</sub> &lt; … &lt; *α*<sub>*J* − 1</sub>.

The likelihood function for the ordinal regression model is

$$ L\_n(\\pmb{\\alpha}, \\pmb{\\beta}) = \\prod\_{i=1}^n \\prod\_{j=1}^J \\left\\{ \\text{logit}^{-1}(\\alpha\_j + \\pmb{x}\_i^T\\pmb{\\beta}) - \\text{logit}^{-1}(\\alpha\_{j-1} + \\pmb{x}\_i^T\\pmb{\\beta} )   \\right\\} ^ {\_(y\_i = j)}. $$

## LASSO Penalization

Let
$l(\\pmb{\\alpha}, \\pmb{\\beta}) = \\frac{-1}{n} \\log L(\\pmb{\\alpha}, \\pmb{\\beta})$
be the standardized log-likelihood.

The LASSO-penalized ordinal regression model is fit by minimizing the
following objective function with respect to **α** and **β**.

$$ f(\\pmb{\\alpha}, \\pmb{\\beta}) = l(\\pmb{\\alpha}, \\pmb{\\beta}) + \\lambda\\sum\_{j=1}^p\|\\beta\_j\|. $$

## Proximal Gradient Descent Algorithm

The objective function can be minimized using a proximal gradient
descent (PGD) algorithm.

Fix the following initial parameters for the PGD algorithm: *m* &gt; 0
(the initial step size), *a* ∈ (0, 1) (the step size decrement value),
and *ϵ* &gt; 0 (the convergence criterion).

The proximal projection operator for the LASSO penalty (applied to **β**
but not to **α**) is

$$ \\text{prox}\_{\\lambda m}(\\pmb{w}, \\pmb{z}) = \\text{argmin}\_{\\pmb{\\alpha}, \\pmb{\\beta}} \\frac{1}{2m} \\left(\|\|\\pmb{w} - \\pmb{\\alpha} \|\|\_2^2 + \|\|\\pmb{z} - \\pmb{\\beta} \|\|\_2^2 \\right) + \\lambda\\sum\_{j=1}^p\|\\beta\_j\| = \\left\\{ \\pmb{w},  \\text{sign}(\\pmb{z})(\\pmb{z} - m \\lambda)\_+ \\right\\} $$

Given current estimates
**θ**<sup>(*k*)</sup> = (**α**<sup>(*k*)</sup>, **β**<sup>(*k*)</sup>)<sup>*T*</sup>,
search for updated estimates **θ**<sup>(*k* + 1)</sup> by following the
steps:

1.  propose a candidate update
    $\\pmb{\\theta} = \\text{prox}\_{\\lambda m}\\left\\{\\pmb{\\theta}^{(k)} - \\frac{1}{m} \\nabla l(\\pmb{\\theta}^{(k)})\\right\\}$,

2.  if the condition
    $l(\\pmb{\\theta}) \\leq l(\\pmb{\\theta}^{(k)}) + \\nabla l(\\pmb{\\theta}^{(k)})^T(\\pmb{\\theta} - \\pmb{\\theta}^{(k)}) + \\frac{1}{2m} (\\pmb{\\theta} - \\pmb{\\theta}^{(k)})^T (\\pmb{\\theta} - \\pmb{\\theta}^{(k)})$
    is met, then make the the update **θ**<sup>(*k* + 1)</sup> = **θ**,

3.  else decrement the step size *m* = *a**m* and returning to step 2.

Continue updating **θ**<sup>(*k*)</sup> until convergence, i.e., until
$\\left\|\\frac{f(\\pmb{\\theta}^{(k+1)}) - f(\\pmb{\\theta}^{(k)})}{f(\\pmb{\\theta}^{(k)})}\\right\| &lt; \\epsilon$.

A technical note is that the **α** parameters are constrained by
*α*<sub>1</sub> &lt; … &lt; *α*<sub>*J* − 1</sub>. We can reparametrize
the model with
**ζ** = (*ζ*<sub>1</sub>, …, *ζ*<sub>*J* − 1</sub>)<sup>*T*</sup>, where
*ζ*<sub>1</sub> = *α*<sub>1</sub> and
*ζ*<sub>*j*</sub> = log (*α*<sub>*j*</sub> − *α*<sub>*j* − 1</sub>) for
*j* = 2, …, *J* − 1. Then **ζ** ∈ ℝ<sup>*J* − 1</sup> have no
constraints. So we can follow the above procedure to minimize the above
objective function with respect to **ζ** and **β**, then back-transform
to obtain estimates for **α**.

## Data Generation

The `HOOVS` package allows the user to simulate their own data with the
`simulate.data()` function. For *n* subjects, we generate *p* covariates
from independent standard normal distributions[1]. Given true parameters
**α**<sub>0</sub> and **β**<sub>0</sub>, we compute the multinomial
probabilities for the outcome for each individual and simulate
*y*<sub>*i*</sub> accordingly.

``` r
# sample size
n <- 1000

# number of covariates
p <- 50

# number of categories for ordinal outcome
J <- 4

# grid of lambdas
lambdas <- seq(0.2, 0, -0.02)

# set population parameters
alpha <- seq(.5, 4, length = J - 1) # category-specific intercepts
beta <- rep(0, p)                     # slope parameters
beta[1: floor(p / 2)] <- 1            # half of the betas are 0, other half are 1

# simulate data according to the above parameters
dat <- simulate.data(
  n = 1000,
  alpha = alpha,
  beta = beta)
```

For this example, we simulated data with *n* = 1000 observations, *p* =
50 covariates, *J* = 4 ordinal outcome categories, and true parameter
values of **α**<sub>0</sub> = (0.5, 2.25, 4) and **β**<sub>0</sub> = (1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0). Note that this implies the first half of the covariates are truly
associated with the outcome and the last half are not. The first 10 rows
and 10 columns of the data set are shown below.

``` r
dat[1:10, 1:10] %>% 
  mutate_if(.predicate = function(x) is.numeric(x),
            .funs = function(x) round(x, digits = 2))
#>    y    X1    X2    X3    X4    X5    X6    X7    X8    X9
#> 1  3 -0.63  1.13 -0.89  0.74 -1.13 -1.52 -0.62 -1.33  0.26
#> 2  3  0.18  1.11 -1.92  0.39  0.76  0.63 -1.11  0.95 -0.83
#> 3  3 -0.84 -0.87  1.62  1.30  0.57 -1.68 -2.17  0.86 -1.46
#> 4  2  1.60  0.21  0.52 -0.80 -1.35  1.18 -0.03  1.06  1.68
#> 5  4  0.33  0.07 -0.06 -1.60 -2.03  1.12 -0.26 -0.35 -1.54
#> 6  1 -0.82 -1.66  0.70  0.93  0.59 -1.24  0.53 -0.13 -0.19
#> 7  3  0.49  0.81  0.05  1.81 -1.41 -1.23 -0.56  0.76  1.02
#> 8  1  0.74 -1.91 -1.31 -0.06  1.61  0.60  1.61 -0.49  0.55
#> 9  1  0.58 -1.25 -2.12  1.89  1.84  0.30  0.56  1.11  0.76
#> 10 1 -0.31  1.00 -0.21  1.58  1.37 -0.11  0.19  1.46 -0.42
```

## Fitting Penalized Model

Now run our version of a LASSO-penalized ordinal regression function on
the simulated data for various values of *λ*: 0.2, 0.18, 0.16, 0.14,
0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.

``` r
# test HOOVS LASSO-penalized ordinal regression function
tic("HOOVS ordreg.lasso() function")
res.ordreg <- ordreg.lasso(
  formula = y ~ .,
  data = dat,
  lambdas = lambdas
)
toc()
#> HOOVS ordreg.lasso() function: 4.283 sec elapsed
coef.ordreg <- cbind(res.ordreg$alpha, res.ordreg$beta)
```

We can now look at how the parameter estimates from our funciton change
as the penalty parameter *λ* changes

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Note in the above plot that the **α** estimates do not shrink all the
way to 0 since they are not penalized in the LASSO model. On the other
hand, the **β** estimates do shrink to 0 as *λ* increases. Recall that
the data were simulated according to a model where half of the **β**
values are truly 0 and the other half are truly 1. It is clear in the
above plot which covariates are truly not associated with the outcome
based on how fast their corresponding parameter estimates shrink to 0.

## Assessing Prediction with Weighted Kappa

We can assess the predictive performance of the model with a weighted
kappa statistic. The following table gives the weighted kappa values for
the models fit using each supplied penalty parameter *λ*.

``` r
data.frame("lambda" = lambdas,
           "Weighted Kappa" = res.ordreg$kappa)
#>             lambda Weighted.Kappa
#> lambda=0.2    0.20     0.00000000
#> lambda=0.18   0.18     0.00000000
#> lambda=0.16   0.16     0.00000000
#> lambda=0.14   0.14     0.00000000
#> lambda=0.12   0.12     0.00000000
#> lambda=0.1    0.10     0.00000000
#> lambda=0.08   0.08     0.02319807
#> lambda=0.06   0.06     0.44565785
#> lambda=0.04   0.04     0.72450154
#> lambda=0.02   0.02     0.83576007
#> lambda=0      0.00     0.90220833
```

## Inference

Suppose we have identified a subset of relevant predictors and want to
perform inference or hypothesis testing using an independent test data
set. We obtain the asymptotic covariance of the parameter estimates from
an ordinal regression model fit with no LASSO penalty by specifying
`return.cov = TRUE`. Then the standard errors of the parameter estimates
are the square roots of the diagonal entries of the covariance matrix.

``` r
# simulate test data
dat <- simulate.data(
  n = 500,
  alpha = alpha,
  beta = beta)

# specify relevant parameters
# (in practice these would be selected using training data)
rel.betas.ind <- which(beta != 0)

# fit model with no penalty
res.ordreg.test <- ordreg.lasso(
  formula = y ~ .,
  data = select(dat, c("y", paste0("X", rel.betas.ind))),
  lambdas = 0,
  return.cov = T
)

# covariance matrix
#res.ordreg.test$cov

# standard error of parameter estimates
sqrt(diag(res.ordreg.test$cov))
#>    alpha1    alpha2    alpha3        X1        X2        X3        X4        X5 
#> 0.1885505 0.2302522 0.3188871 0.1518158 0.1493560 0.1437412 0.1492866 0.1425663 
#>        X6        X7        X8        X9       X10       X11       X12       X13 
#> 0.1532874 0.1598190 0.1501358 0.1533953 0.1471709 0.1463651 0.1503864 0.1473245 
#>       X14       X15       X16       X17       X18       X19       X20       X21 
#> 0.1385858 0.1596688 0.1555481 0.1417651 0.1514717 0.1404234 0.1415255 0.1427013 
#>       X22       X23       X24       X25 
#> 0.1532489 0.1518607 0.1467264 0.1558101
```

# Method 2: Random Forest

[1] the current version can only handle continuous covariates, and
categorical variables must be created into dummy variables by hand.
