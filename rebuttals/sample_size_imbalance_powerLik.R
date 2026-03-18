# ============================================================
# 0. Libraries
# ============================================================
if (!requireNamespace("mvtnorm", quietly = TRUE)) install.packages("mvtnorm")
if (!requireNamespace("loo", quietly = TRUE)) install.packages("loo")
library(mvtnorm)
library(loo)
library(ManyData)
library(causl)
library(data.table)
library(ggplot2)

set.seed(42)

# ============================================================
# 1. Load dataset
# ============================================================
all_data <- read.csv("/Users/evangelosdimitriou/myGitRepo/CausalICM/CLeaR/rebuttals/diff_sample_size_data.csv")
head(all_data)

# ============================================================
# 2. Helper functions (from your previous workflow)
# ============================================================

merge_formulas <- function(formulas) list(formula = formulas[[1]], wh = list(beta = TRUE, phi = TRUE))

masks <- function(formulas, family, wh) {
  beta_names <- names(coef(lm(formulas[[1]], data = dat_e)))
  list(beta_m = setNames(rep(1, length(beta_names)), beta_names), phi_m = 1)
}

nll2 <- function(theta, dat, mm = NULL, mask_beta, mask_phi, seqphi, fam_cop,
                 family, link, useC) {
  X <- model.matrix(Y ~ A * poly(X, 2, raw = TRUE), data = dat)
  beta <- theta[1:ncol(X)]
  mu <- as.vector(X %*% beta)
  -sum(dnorm(dat$Y, mu, 1, log = TRUE))
}

lhs <- function(formulas) c("Y")

llC <- function(y, mm, beta_m, phi, inCop) {
  mu <- as.vector(mm %*% beta_m)
  dnorm(y, mu, 1, log = TRUE)
}

ApproxFI_single <- function(msk, theta, mm, dat, delta) diag(length(theta))

ManyData <- list2env(list(ApproxFI_single = ApproxFI_single, llC = llC))
causl <- list2env(list(nll2 = nll2, lhs = lhs))

# ============================================================
# 3. Function to compute ELPD for a given eta
# ============================================================

compute_elpd <- function(dat_e, dat_o, eta, mcmc_pars) {
  formulas <- list(Y ~ A * poly(X, 2, raw = TRUE), ~ A * poly(X, 2, raw = TRUE), ~ 1)
  family <- list(gaussian(), binomial(), gaussian())
  
  start <- rep(0, ncol(model.matrix(Y ~ A * poly(X, 2, raw = TRUE), data = dat_e)))
  msks <- list(obs = masks(formulas[-2], family[-2], NULL),
               exp = masks(formulas[-2], family[-2], NULL))
  
  theta_curr <- start
  prop_sigma <- diag(length(start)) * 0.01
  chain <- matrix(NA, nrow = (mcmc_pars$n_iter - mcmc_pars$n_burn) / mcmc_pars$n_thin, ncol = length(start))
  rec <- 0
  
  curr_ll <- -causl$nll2(theta_curr, dat_e, NULL, msks$exp$beta_m, msks$exp$phi_m, NULL, 1, family, NULL, TRUE) +
    -eta * causl$nll2(theta_curr, dat_o, NULL, msks$obs$beta_m, msks$obs$phi_m, NULL, 1, family, NULL, TRUE)
  
  for (i in seq_len(mcmc_pars$n_iter)) {
    theta_prop <- theta_curr + mvtnorm::rmvnorm(1, sigma = prop_sigma)
    prop_ll <- -causl$nll2(theta_prop, dat_e, NULL, msks$exp$beta_m, msks$exp$phi_m, NULL, 1, family, NULL, TRUE) +
      -eta * causl$nll2(theta_prop, dat_o, NULL, msks$obs$beta_m, msks$obs$phi_m, NULL, 1, family, NULL, TRUE)
    if (-rexp(1) < (prop_ll - curr_ll)) {
      theta_curr <- theta_prop
      curr_ll <- prop_ll
    }
    if (i > mcmc_pars$n_burn && ((i - mcmc_pars$n_burn - 1) %% mcmc_pars$n_thin == 0)) {
      rec <- rec + 1
      chain[rec, ] <- theta_curr
    }
  }
  
  mm_exp <- model.matrix(formulas[[1]], data = dat_e)
  out <- sapply(1:nrow(chain), function(i) ManyData$llC(dat_e[, lhs(formulas)[1]], mm_exp, chain[i, ], 1, NULL))
  waic_eta <- loo::waic(out)
  waic_eta$estimates["elpd_waic", "Estimate"]
}

# ============================================================
# 4. Compute eta_optim across all datasets
# ============================================================

mcmc_pars <- list(n_iter = 50, n_burn = 10, n_thin = 2)
eta_grid <- seq(0, 1, by = 0.2)
dataset_ids <- sort(unique(all_data$datasetNo))

elpd_results <- data.frame(eta = eta_grid, elpd_mean = NA_real_, elpd_sd = NA_real_)

for (eta in eta_grid) {
  elpds <- c()
  for (d in dataset_ids) {
    dat_d <- subset(all_data, datasetNo == d)
    dat_e <- subset(dat_d, S == 1)
    dat_o <- subset(dat_d, S == 0)
    elpds <- c(elpds, compute_elpd(dat_e, dat_o, eta, mcmc_pars))
  }
  elpd_results$elpd_mean[elpd_results$eta == eta] <- mean(elpds)
  elpd_results$elpd_sd[elpd_results$eta == eta] <- sd(elpds) / sqrt(length(elpds))
}

# Plot for visual check
ggplot(elpd_results, aes(x = eta, y = elpd_mean)) +
  geom_point(size = 3) +
  geom_line() +
  geom_errorbar(aes(ymin = elpd_mean - elpd_sd, ymax = elpd_mean + elpd_sd), width = 0.05) +
  labs(x = expression(eta), y = "Average ELPD (WAIC)",
       title = "Average ELPD vs Eta across datasets") +
  theme_minimal()

# Optimal eta
eta_optim <- elpd_results$eta[which.max(elpd_results$elpd_mean)]
cat("Optimal eta:", eta_optim, "\n")

# ============================================================
# 5. Train causal model and compute RMSE for each sample size
# ============================================================

sample_sizes <- sort(unique(all_data$obs_sample_size))
x_grid <- seq(-2, 2, length.out = 50)
tau_true <- function(x) 1 + x + x^2

final_results <- list()

for (ss in sample_sizes) {
  cat("Processing sample size:", ss, "\n")
  
  datasets_ss <- unique(all_data$datasetNo)
  rmse_list <- numeric(length(datasets_ss))
  cate_estimates <- list()
  beta_means <- vector("list", length(datasets_ss))
  
  for (i in seq_along(datasets_ss)) {
    dat_i <- subset(all_data, datasetNo == datasets_ss[i] & obs_sample_size == ss)
    exp_data <- subset(dat_i, S == 1)
    obs_data <- subset(dat_i, S == 0)
    
    # Fit causal models
    forms_exp <- list(Y ~ A * (X + I(X^2)), A ~ X, cop ~ A + X)
    forms_obs <- list(Y ~ A * (X + I(X^2)), A ~ X, cop ~ A + X)
    
    fit_exp <- fit_causl(dat = exp_data, formulas = causl:::tidy_formulas(forms_exp, kwd = "cop"), family = list(1,1,1))
    fit_obs <- fit_causl(dat = obs_data, formulas = causl:::tidy_formulas(forms_obs, kwd = "cop"), family = list(1,1,1))
    
    # Posterior
    samples <- approx_posterior(fit_exp, fit_obs, eta_optim, n_sample = 2000)
    posterior_means <- colMeans(samples)
    beta_means[[i]] <- posterior_means[c("beta_A", "beta_AX", "beta_AX2")]
    
    # Estimated CATE
    CATE_est <- posterior_means["beta_A"] + posterior_means["beta_AX"] * x_grid + posterior_means["beta_AX2"] * x_grid^2
    CATE_true_vals <- tau_true(x_grid)
    
    # RMSE
    rmse_list[i] <- sqrt(mean((CATE_est - CATE_true_vals)^2))
    
    # Store for plotting
    cate_estimates[[i]] <- data.table(dataset = datasets_ss[i], x = x_grid, CATE_est = CATE_est, CATE_true = CATE_true_vals)
  }
  
  final_results[[paste0("sample_size_", ss)]] <- list(
    rmse_list = rmse_list,
    cate_estimates = cate_estimates,
    beta_means = beta_means,
    mean_rmse = mean(rmse_list)
  )
  
  cat("Mean RMSE for sample size", ss, ":", mean(rmse_list), "\n")
}

# ============================================================
# 6. Save results
# ============================================================

rmse_records <- data.frame(
  obs_sample_size = integer(),
  datasetNo = integer(),
  rmse = numeric()
)

for (ss in names(final_results)) {
  # Extract numeric sample size
  ss_num <- as.numeric(gsub("sample_size_", "", ss))
  
  rmse_list <- final_results[[ss]]$rmse_list
  datasets_ss <- unique(all_data$datasetNo)
  
  df_ss <- data.frame(
    obs_sample_size = ss_num,
    datasetNo = datasets_ss,
    rmse = rmse_list
  )
  
  rmse_records <- rbind(rmse_records, df_ss)
}

# Save to CSV
write.csv(rmse_records, "/Users/evangelosdimitriou/myGitRepo/CausalICM/CLeaR/rebuttals/sample_size_imbalanxe_powerLik_results.csv", row.names = FALSE)
