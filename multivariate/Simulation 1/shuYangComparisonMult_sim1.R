# ================================
# User-defined path
# ================================
# 👉 IMPORTANT: before running the script, the user must set:
# base_path <- "/path/to/your/CausalICM/project/"
# The folder must contain:
#   - multivariate/Simulation 1/datasets_sim1_mult.csv
#   - (other outputs will be saved here)
# ================================

# Example:
# base_path <- "/Users/username/Research/CausalICM/"
# ================================

# Load dataset
datasets_sim1_multi <- read.csv(
  paste0(base_path, "multivariate/Simulation 1/datasets_sim1_mult.csv")
)

# Install and load Integrative HTE package
devtools::install_github("shuyang1987/IntegrativeHTEcf")

library(mgcv)
library(MASS)
library(rootSolve)

# Generate X_test grid
X_test <- cbind(
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04),
  seq(-2.0, 2.04, by = 0.04)
)

# Initialise vectors
rmse_values <- numeric(100)
bias_values <- numeric(100)
variance_values <- numeric(100)

bias_shuYang_df <- data.frame("X_test" = X_test)

# ================================
# Main loop
# ================================
for (i in 1:100) {
  print(i)
  
  data_rct <- datasets_sim1_multi[datasets_sim1_multi$datasetNo == i & datasets_sim1_multi$S == 1, ]
  data_obs <- datasets_sim1_multi[datasets_sim1_multi$datasetNo == i & datasets_sim1_multi$S == 0, ]
  
  A <- c(data_rct$A, data_obs$A)
  X <- rbind(data_rct[, c("X1", "X2", "X3", "X4", "X5")],
             data_obs[, c("X1", "X2", "X3", "X4", "X5")])
  
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))
  Y <- c(data_rct$Y, data_obs$Y)
  
  X.hte <- cbind(X, X^2)
  X.cf  <- X
  
  intHTE <- IntegrativeHTEcf::IntHTEcf(
    A, X, X.hte, X.cf, Y, S, nboots = 50
  )
  
  # True treatment effect
  true_tau <- 1 + X_test[, 1] + X_test[, 2]
  
  # Predicted treatment effect
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
  
  # Store performance metrics
  rmse_values[i]    <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values[i]    <- mean(true_tau - pred_tau)
  variance_values[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  bias_shuYang_df[[paste0("Iteration_", i)]] <- true_tau - pred_tau
}

# ================================
# Export results
# ================================

results_shuYang_sim1_multi <- data.frame(
  RMSE     = rmse_values,
  Bias     = bias_values,
  Variance = variance_values
)

write.csv(
  results_shuYang_sim1_multi,
  file = paste0(base_path, "multivariate/Simulation 1/results_ShuYang.csv"),
  row.names = TRUE
)

write.csv(
  bias_shuYang_df,
  file = paste0(base_path, "multivariate/Simulation 1/bias_ShuYang.csv"),
  row.names = FALSE
)
