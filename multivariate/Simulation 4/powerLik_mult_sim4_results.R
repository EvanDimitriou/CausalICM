# ============================================================
# ELPD across datasets and eta values
# ============================================================

# --- 0. Libraries
if (!requireNamespace("mvtnorm", quietly = TRUE)) install.packages("mvtnorm")
if (!requireNamespace("loo", quietly = TRUE)) install.packages("loo")
library(mvtnorm)
library(loo)
library(ManyData)
library(causl)
library(data.table)

set.seed(42)

# ============================================================
# 1. Load datasets
# ============================================================
all_data <- read.csv("datasets_sim4_mult.csv")  # anonymised path
head(all_data)

# ============================================================
# 2. Helper functions
# ============================================================
merge_formulas <- function(formulas) {
  list(formula = formulas[[1]], wh = list(beta = TRUE, phi = TRUE))
}

masks <- function(formulas, family, wh) {
  beta_names <- names(coef(lm(formulas[[1]], data = dat_e)))
  list(beta_m = setNames(rep(1, length(beta_names)), beta_names),
       phi_m = 1)
}

nll2 <- function(theta, dat, mm = NULL, mask_beta, mask_phi, seqphi, fam_cop,
                 family, link, useC, formula = NULL) {
  if (!is.null(formula)) {
    X <- model.matrix(formula, data = dat)
  } else {
    if (all(c("X1", "X2") %in% names(dat))) {
      X <- model.matrix(Y ~ A * (poly(X1, 2, raw = TRUE) + poly(X2, 2, raw = TRUE)), data = dat)
    } else if ("X" %in% names(dat)) {
      X <- model.matrix(Y ~ A * poly(X, 2, raw = TRUE), data = dat)
    } else {
      stop("No appropriate covariates found in dat.")
    }
  }
  beta <- theta[1:ncol(X)]
  mu <- as.vector(X %*% beta)
  -sum(dnorm(dat$Y, mu, 1, log = TRUE))
}

lhs <- function(formulas) c("Y")

llC <- function(y, mm, beta_m, phi, inCop) {
  mu <- as.vector(mm %*% beta_m)
  dnorm(y, mu, 1, log = TRUE)
}

ApproxFI_single <- function(msk, theta, mm, dat, delta) {
  p <- length(theta)
  diag(rep(1, p))
}

ManyData <- list2env(list(ApproxFI_single = ApproxFI_single, llC = llC))
causl <- list2env(list(nll2 = nll2, lhs = lhs))

# ============================================================
# 3. MCMC + ELPD core function
# ============================================================
compute_elpd <- function(dat_e, dat_o, eta, mcmc_pars) {
  formulas <- list(
    Y ~ A * (poly(X1, 2, raw = TRUE) + poly(X2, 2, raw = TRUE)),
    ~ A * (poly(X1, 2, raw = TRUE) + poly(X2, 2, raw = TRUE)),
    ~ 1
  )
  family <- list(gaussian(), binomial(), gaussian())
  
  start <- rep(0, ncol(model.matrix(formulas[[1]], data = dat_e)))
  
  msks <- list(obs = masks(formulas[-2], family[-2], NULL),
               exp = masks(formulas[-2], family[-2], NULL))
  
  theta <- list(obs = start, exp = start)
  FI_exp <- ManyData$ApproxFI_single(msks$exp, theta$exp, NULL, dat_e, delta = 1e-4)
  FI_obs <- ManyData$ApproxFI_single(msks$obs, theta$obs, NULL, dat_o, delta = 1e-4)
  prop_sigma <- 400 * solve(10 * FI_exp + eta * FI_obs)
  
  theta_curr <- start
  curr_ll <- -causl$nll2(theta_curr, dat_e, NULL, msks$exp$beta_m, msks$exp$phi_m, NULL,
                         1, family, NULL, TRUE, formula = formulas[[1]]) +
    -eta * causl$nll2(theta_curr, dat_o, NULL, msks$obs$beta_m, msks$obs$phi_m,
                      NULL, 1, family, NULL, TRUE, formula = formulas[[1]])
  
  chain <- matrix(NA, nrow = (mcmc_pars$n_iter - mcmc_pars$n_burn) / mcmc_pars$n_thin,
                  ncol = length(theta_curr))
  rec <- 0
  
  for (i in seq_len(mcmc_pars$n_iter)) {
    theta_prop <- theta_curr + mvtnorm::rmvnorm(1, sigma = prop_sigma / 100)
    prop_ll <- -causl$nll2(theta_prop, dat_e, NULL, msks$exp$beta_m, msks$exp$phi_m, NULL,
                           1, family, NULL, TRUE, formula = formulas[[1]]) +
      -eta * causl$nll2(theta_prop, dat_o, NULL, msks$obs$beta_m, msks$obs$phi_m,
                        NULL, 1, family, NULL, TRUE, formula = formulas[[1]])
    if (-rexp(1) < (prop_ll - curr_ll)) {
      theta_curr <- theta_prop
      curr_ll <- prop_ll
    }
    if (i > mcmc_pars$n_burn && ((i - mcmc_pars$n_burn - 1) %% mcmc_pars$n_thin == 0)) {
      rec <- rec + 1
      chain[rec, ] <- theta_curr
    }
  }
  
  theta_sim_rtnd <- chain
  vars <- causl$lhs(formulas[c(1, 3)])
  mm_exp <- model.matrix(formulas[[1]], data = dat_e)
  out <- matrix(NA, nrow = nrow(theta_sim_rtnd), ncol = nrow(dat_e))
  
  for (i in seq_len(nrow(theta_sim_rtnd))) {
    beta_vec <- theta_sim_rtnd[i, 1:ncol(mm_exp)]
    out[i, ] <- ManyData$llC(dat_e[, vars], mm_exp, beta_vec, 1, c(1, 2))
  }
  
  waic_eta <- loo::waic(out)
  return(waic_eta$estimates["elpd_waic", "Estimate"])
}

# ============================================================
# 4. Main loop over eta and datasets
# ============================================================
mcmc_pars <- list(n_iter = 50, n_burn = 10, n_thin = 2)
eta_grid <- seq(0, 1, by = 0.2)
dataset_ids <- sort(unique(all_data$datasetNo))

results <- data.frame(eta = eta_grid, elpd_mean = NA_real_, elpd_sd = NA_real_)

for (eta in eta_grid) {
  elpds <- c()
  cat("\n========================\nETA =", eta, "\n========================\n")
  
  for (d in dataset_ids) {
    dat_d <- subset(all_data, datasetNo == d)
    dat_e <- subset(dat_d, S == 1)
    dat_o <- subset(dat_d, S == 0)
    
    elpds <- c(elpds, compute_elpd(dat_e, dat_o, eta, mcmc_pars))
    cat("Dataset", d, "ELPD =", tail(elpds, 1), "\n")
  }
  
  results$elpd_mean[results$eta == eta] <- mean(elpds)
  results$elpd_sd[results$eta == eta]  <- sd(elpds) / sqrt(length(elpds))
  
  cat("Average ELPD for eta =", eta, ":", 
      mean(elpds), 
      "SE =", results$elpd_sd[results$eta == eta], "\n")
}

# ============================================================
# 5. Plot ELPD vs eta
# ============================================================
library(ggplot2)

ggplot(results, aes(x = eta, y = elpd_mean)) +
  geom_point(size = 3) +
  geom_line() +
  geom_errorbar(aes(ymin = elpd_mean - elpd_sd, ymax = elpd_mean + elpd_sd), width = 0.05) +
  labs(x = expression(eta), y = "Average ELPD (WAIC)",
       title = "Average ELPD vs Eta across datasets") +
  theme_minimal()

eta_optim <- results$eta[which.max(results$elpd_mean)]
cat("Optimal eta =", eta_optim, "\n")

# ============================================================
# 6. Approx posterior & CATE estimation
# ============================================================
eta <- eta_optim
n_sample <- 2000

x1_grid <- seq(-2, 2, length.out = 50)
x2_grid <- seq(-2, 2, length.out = 50)
grid <- expand.grid(X1 = x1_grid, X2 = x2_grid)

tau_true <- function(x1, x2) 1 + x1 + x1^2 + x2 + x2^2

approx_posterior <- function(fit_exp, fit_obs, eta, n_sample) {
  theta_hat <- fit_exp$par + eta * (fit_obs$par - fit_exp$par)
  FI <- fit_exp$sandwich + eta * fit_obs$sandwich
  invFI <- tryCatch(solve(FI), error = function(e) MASS::ginv(FI))
  
  samples <- MASS::mvrnorm(n = n_sample, mu = theta_hat, Sigma = invFI)
  if (is.vector(samples)) samples <- matrix(samples, nrow = n_sample, byrow = TRUE)
  
  colnames(samples) <- c(
    "beta_0", "beta_A",
    "beta_X1", "beta_X2",
    "beta_X1_2", "beta_X2_2",
    "beta_AX1", "beta_AX2",
    "beta_AX1_2", "beta_AX2_2",
    "alpha_0", "alpha_X1", "alpha_X2",
    "gamma_1", "gamma_2"
  )
  
  return(samples)
}

datasets <- unique(all_data$datasetNo)
rmse_list <- numeric(length(datasets))
cate_estimates <- list()
beta_means <- vector("list", length(datasets))

for (i in seq_along(datasets)) {
  dat_i <- subset(all_data, datasetNo == datasets[i])
  exp_data <- subset(dat_i, S == 1)
  obs_data <- subset(dat_i, S == 0)
  
  forms_exp <- list(
    Y ~ A * (X1 + X2 + I(X1^2) + I(X2^2)),
    A ~ X1 + X2,
    cop ~ A + X1 + X2
  )
  
  forms_obs <- forms_exp
  family_exp <- list(1, 1, 1)
  family_obs <- list(1, 1, 1)
  
  forms2 <- list(
    exp = causl:::tidy_formulas(forms_exp, kwd = "cop"),
    obs = causl:::tidy_formulas(forms_obs, kwd = "cop")
  )
  
  fit_exp <- fit_causl(dat = exp_data, formulas = forms2$exp, family = family_exp)
  fit_obs <- fit_causl(dat = obs_data, formulas = forms2$obs, family = family_obs)
  
  samples <- approx_posterior(fit_exp, fit_obs, eta, n_sample)
  posterior_means <- colMeans(samples)
  
  beta_means[[i]] <- posterior_means[
    c("beta_A", "beta_AX1", "beta_AX2", "beta_AX1_2", "beta_AX2_2")
  ]
  
  CATE_est <- with(grid,
                   posterior_means["beta_A"] +
                     posterior_means["beta_AX1"]   * X1 +
                     posterior_means["beta_AX2"]   * X2 +
                     posterior_means["beta_AX1_2"] * X1^2 +
                     posterior_means["beta_AX2_2"] * X2^2)
  
  CATE_true <- tau_true(grid$X1, grid$X2)
  rmse_list[i] <- sqrt(mean((CATE_est - CATE_true)^2))
  
  cate_estimates[[i]] <- data.table(
    dataset = datasets[i],
    X1 = grid$X1,
    X2 = grid$X2,
    CATE_est = CATE_est,
    CATE_true = CATE_true
  )
  
  cat("Finished dataset", datasets[i],
      "with RMSE =", round(rmse_list[i], 3), "\n")
}

mean_rmse <- mean(rmse_list)
sd_rmse <- sd(rmse_list)
cat("\nAverage RMSE across datasets:", round(mean_rmse, 3), "\n")
cat("SD of RMSE:", round(sd_rmse, 3), "\n")

# ============================================================
# 7. Save results
# ============================================================
saveRDS(list(
  eta = eta,
  rmse_list = rmse_list,
  mean_rmse = mean_rmse,
  sd_rmse = sd_rmse,
  betas_list = beta_means
), file = "cate_rmse_results_mult_sim4.rds")
