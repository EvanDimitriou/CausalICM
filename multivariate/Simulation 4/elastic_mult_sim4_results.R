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

full_data <- read.csv("datasets_sim4_mult.csv")  # generic path
full_data <- cbind(full_data, q = 1)            # add constant column

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
    X1 = dat_rct$X1,
    X2 = dat_rct$X2,
    X3 = dat_rct$X3,
    X4 = dat_rct$X4,
    X5 = dat_rct$X5,
    q = dat_rct$q,
    ps = dat_rct$ps,
    ml.ps = dat_rct$ml.ps
  )
  
  dat_rwd <- data.frame(
    Y = dat_rwd$Y,
    A = dat_rwd$A,
    X1 = dat_rwd$X1,
    X2 = dat_rwd$X2,
    X3 = dat_rwd$X3,
    X4 = dat_rwd$X4,
    X5 = dat_rwd$X5,
    q = dat_rwd$q
  )
  
  # Convert to matrices for ElasticIntegrative
  Data.list <- list(
    RT = dat_rct,
    RW = dat_rwd
  )
  
  for (i in 1:5) {
    Data.list$RT[[paste0("X", i)]] <- as.matrix(Data.list$RT[[paste0("X", i)]])
    Data.list$RW[[paste0("X", i)]] <- as.matrix(Data.list$RW[[paste0("X", i)]])
  }
  
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
    mainName = c("X1", "X2"),
    contName = c("X1", "X2"),
    propenName = c("X1", "X2"),
    fixed = FALSE,
    nboot = 4
  )
  
  # Store results
  results_list[[paste0("dataset_", ds)]] <- list(adaptive = res_adaptive)
  cat("Processed dataset", ds, "\n")
}

# Save results
saveRDS(results_list, file = "all_results_mult_sim4.rds")

# ============================================================
# 5. Compute RMSEs across datasets
# ============================================================

compute_RMSE <- function(results_list) {
  
  # Sequences for each dimension
  x1_grid <- seq(-2, 2, length.out = 100)
  x2_grid <- seq(-2, 2, length.out = 100)
  
  # True CATE function
  tau_true <- function(x1, x2) 1 + x1 + x1^2 + x2 + x2^2
  true_vals <- tau_true(x1_grid, x2_grid)
  
  crit <- qchisq(0.95, df = 1)
  rmse_list <- list()
  
  for (nm in names(results_list)) {
    res <- results_list[[nm]]$adaptive
    Tstat <- res$nuispar["Tstat.psi"]
    
    # Select estimator
    if (Tstat < crit) {
      beta0 <- res$est["elas.1"]
      beta1 <- res$est["elas.2"]
      beta2 <- res$est["elas.3"]
    } else {
      beta0 <- res$est["ee.rt(ml).1"]
      beta1 <- res$est["ee.rt(ml).2"]
      beta2 <- res$est["ee.rt(ml).3"]
    }
    
    # Compute estimated CATE
    est_vals <- beta0 + beta1 * x1_grid + beta2 * x2_grid
    rmse <- sqrt(mean((est_vals - true_vals)^2))
    rmse_list[[nm]] <- rmse
  }
  
  return(unlist(rmse_list))
}

rmse_results_mult_sim4 <- compute_RMSE(results_list)
saveRDS(rmse_results_mult_sim4, file = "mult_sim4_rmse.rds")
boxplot(rmse_results_mult_sim4)
