# ============================================================
# 1. Load required libraries
# ============================================================

library(ElasticIntegrative)
library(SuperLearner)
library(rootSolve)
library(ggplot2)

# ============================================================
# 2. Load dataset
# ============================================================

full_data <- read.csv("datasets_sim1.csv")  # generic path
full_data <- cbind(full_data, q = 1)       # add constant column

results_list <- list()
dataset_ids <- unique(full_data$datasetNo)

# ============================================================
# 3. Loop through datasets
# ============================================================

for (ds in dataset_ids) {
  
  # Subset dataset
  dat_subset <- subset(full_data, datasetNo == ds)
  
  # Split into RCT and observational
  dat_rct <- subset(dat_subset, S == 1)
  dat_rct <- cbind(dat_rct, ps = 0.5, ml.ps = 0.5)
  
  dat_rwd <- subset(dat_subset, S == 0)
  
  # Keep relevant columns
  dat_rct <- data.frame(
    Y = dat_rct$Y,
    A = dat_rct$A,
    X1 = dat_rct$X,
    q = dat_rct$q,
    ps = dat_rct$ps,
    ml.ps = dat_rct$ml.ps
  )
  
  dat_rwd <- data.frame(
    Y = dat_rwd$Y,
    A = dat_rwd$A,
    X1 = dat_rwd$X,
    q = dat_rwd$q
  )
  
  # Convert to matrices for ElasticIntegrative
  Data.list <- list(
    RT = dat_rct,
    RW = dat_rwd
  )
  
  Data.list$RT$X1 <- as.matrix(Data.list$RT$X1)
  Data.list$RW$X1 <- as.matrix(Data.list$RW$X1)
  Data.list$RT$Y <- as.matrix(Data.list$RT$Y)
  Data.list$RW$Y <- as.matrix(Data.list$RW$Y)
  Data.list$RT$A <- as.matrix(Data.list$RT$A)
  Data.list$RW$A <- as.matrix(Data.list$RW$A)
  
  # ============================================================
  # 4. Run elasticHTE (adaptive)
  # ============================================================
  
  res_adaptive <- elasticHTE(
    dat.t = Data.list$RT,
    dat.os = Data.list$RW,
    mainName = "X1",
    contName = "X1",
    propenName = "X1",
    fixed = FALSE,
    nboot = 5
  )
  
  # Store results
  results_list[[paste0("dataset_", ds)]] <- list(adaptive = res_adaptive)
  cat("Processed dataset", ds, "\n")
}

# Save results
saveRDS(results_list, file = "all_results_sim1.rds")

# ============================================================
# 5. Example: plot estimated CATE vs X1 for dataset 1
# ============================================================

x_grid <- seq(-2, 2, length.out = 100)
tau_hat <- results_list$dataset_1$adaptive$est["elas.1"] +
  results_list$dataset_1$adaptive$est["elas.2"] * x_grid

plot(x_grid, tau_hat, type = "l", lwd = 2,
     xlab = "X1", ylab = "Estimated CATE",
     main = "Conditional Average Treatment Effect vs X1")

# ============================================================
# 6. Compute RMSEs across datasets
# ============================================================

compute_RMSE <- function(results_list) {
  
  x_grid <- seq(-2, 2, length.out = 200)
  tau_true <- function(x) 1 + x
  true_vals <- tau_true(x_grid)
  crit <- qchisq(0.95, df = 1)
  
  rmse_list <- list()
  
  for (nm in names(results_list)) {
    res <- results_list[[nm]]$adaptive
    Tstat <- res$nuispar["Tstat.psi"]
    
    if (Tstat < crit) {
      beta0 <- res$est["elas.1"]
      beta1 <- res$est["elas.2"]
    } else {
      beta0 <- res$est["ee.rt(ml).1"]
      beta1 <- res$est["ee.rt(ml).2"]
    }
    
    est_vals <- beta0 + beta1 * x_grid
    rmse <- sqrt(mean((est_vals - true_vals)^2))
    rmse_list[[nm]] <- rmse
  }
  
  return(unlist(rmse_list))
}

rmse_results <- compute_RMSE(results_list)
saveRDS(rmse_results, file = "sim1_rmse.rds")
boxplot(rmse_results)
