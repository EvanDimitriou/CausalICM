############################################################
# INSTRUCTIONS FOR USER
# ---------------------
# 1. Set BASE_PATH to the directory where your CausalICM folder lives.
#       Example:
#       BASE_PATH <- "/Users/username/projects/CausalICM"
#
# 2. Required directory structure:
#       CausalICM/
#           multivariate/
#               Simulation_4/
#                   datasets_sim4_mult.csv
#
# 3. This script runs the Integrative HTE model 100 times
#    and saves RMSE, bias, variance, and bias trajectories.
############################################################

# ==========================
# 1. USER-DEFINED BASE PATH
# ==========================
BASE_PATH <- "/path/to/CausalICM"   # <-- EDIT THIS BEFORE RUNNING

sim4_path <- file.path(BASE_PATH, "multivariate", "Simulation_4")

# ==========================
# 2. Load dataset
# ==========================
datasets_sim4_multi <- read.csv(
  file.path(sim4_path, "datasets_sim4_mult.csv")
)

# ==========================
# 3. Install + load packages
# ==========================
# install.packages("devtools")
# devtools::install_github("shuyang1987/IntegrativeHTEcf")

library(IntegrativeHTEcf)
library(mgcv)
library(MASS)
library(rootSolve)

# ==========================
# 4. Test points
# ==========================
X_test <- cbind(
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04)
)

# ==========================
# 5. Storage objects
# ==========================
rmse_values <- numeric(100)
bias_values <- numeric(100)
variance_values <- numeric(100)

bias_shuYang_df <- data.frame(X_test = X_test)

# ==========================
# 6. Main simulation loop
# ==========================
for (i in 1:100) {
  cat("Iteration:", i, "\n")
  
  data_rct <- datasets_sim4_multi[datasets_sim4_multi$datasetNo == i &
                                    datasets_sim4_multi$S == 1, ]
  
  data_obs <- datasets_sim4_multi[datasets_sim4_multi$datasetNo == i &
                                    datasets_sim4_multi$S == 0, ]
  
  # Extract variables
  A1 <- data_rct$A
  A2 <- data_obs$A
  X1 <- data_rct[, c("X1", "X2", "X3", "X4", "X5")]
  X2 <- data_obs[, c("X1", "X2", "X3", "X4", "X5")]
  
  A <- c(A1, A2)
  X <- rbind(X1, X2)
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))
  Y <- c(data_rct$Y, data_obs$Y)
  
  # Covariates for model
  X.hte <- cbind(X, X^2)
  X.cf <- X
  
  X <- as.matrix(X)
  X.hte <- as.matrix(X.hte)
  X.cf <- as.matrix(X.cf)
  
  # Fit Integrative HTE
  intHTE <- IntHTEcf(A, X, X.hte, X.cf, Y, S, nboots = 50)
  
  # True CATE
  true_tau <- 1 +
    X_test[, 1] + X_test[, 1]^2 +
    X_test[, 2] + X_test[, 2]^2
  
  # Predicted CATE
  pred_tau <- intHTE$est.int[1] +
    intHTE$est.int[2] * X_test[, 1] +
    intHTE$est.int[3] * X_test[, 2] +
    intHTE$est.int[4] * X_test[, 3] +
    intHTE$est.int[5] * X_test[, 4] +
    intHTE$est.int[6] * X_test[, 5] +
    intHTE$est.int[7] * X_test[, 1]^2 +
    intHTE$est.int[8] * X_test[, 2]^2 +
    intHTE$est.int[9] * X_test[, 3]^2 +
    intHTE$est.int[10] * X_test[, 4]^2 +
    intHTE$est.int[11] * X_test[, 5]^2
  
  # Metrics
  rmse_values[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values[i] <- mean(true_tau - pred_tau)
  variance_values[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  bias_shuYang_df[[paste0("Iteration_", i)]] <- true_tau - pred_tau
}

# ==========================
# 7. Save results
# ==========================
results_shuYang_sim4_multi <- data.frame(
  RMSE = rmse_values,
  Bias = bias_values,
  Variance = variance_values
)

write.csv(
  results_shuYang_sim4_multi,
  file = file.path(sim4_path, "results_ShuYang.csv"),
  row.names = TRUE
)

write.csv(
  bias_shuYang_df,
  file = file.path(sim4_path, "bias_ShuYang.csv"),
  row.names = FALSE
)
