# -------------------------------
# Load required libraries
# -------------------------------
# Install IntegrativeHTEcf if not already installed
# devtools::install_github("shuyang1987/IntegrativeHTEcf")

library(mgcv)
library(MASS)
library(rootSolve)
library(IntegrativeHTEcf)

# -------------------------------
# Load simulation dataset
# -------------------------------
# Replace the path with your local CausalICM folder
datasets_sim1 <- read.csv("/path/to/your/CausalICM/Simulation1/datasets_sim1.csv")

# -------------------------------
# Set up test data and results storage
# -------------------------------
X_test <- seq(-2.0, 2.0, by = 0.04)

n_iterations <- 100
rmse_values <- numeric(n_iterations)
bias_values <- numeric(n_iterations)
variance_values <- numeric(n_iterations)

bias_shuYang_df <- data.frame("X_test" = X_test)

# -------------------------------
# Main loop: Fit Integrative HTE
# -------------------------------
for (i in 1:n_iterations) {
  print(paste("Iteration", i))
  
  # Subset RCT and observational datasets
  data_rct <- datasets_sim1[datasets_sim1$datasetNo == i & datasets_sim1$S == 1, ]
  data_obs <- datasets_sim1[datasets_sim1$datasetNo == i & datasets_sim1$S == 0, ]
  
  # Extract treatment and covariates
  A1 <- data_rct$A
  X1 <- data_rct$X
  A2 <- data_obs$A
  X2 <- data_obs$X
  
  # Combine RCT and observational data
  A <- c(A1, A2)
  X <- c(X1, X2)
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))  # participation indicator
  Y <- c(data_rct$Y, data_obs$Y)
  
  # Covariate matrices for HTE model
  X.hte <- as.matrix(cbind(X, X^2))
  X.cf <- as.matrix(X)
  X <- as.matrix(X)
  
  # Fit Integrative HTE model
  intHTE <- IntegrativeHTEcf::IntHTEcf(A, X, X.hte, X.cf, Y, S, nboots = 50)
  
  # True treatment effect (CATE)
  true_tau <- 1 + X_test
  
  # Predicted treatment effect using model coefficients
  pred_tau <- intHTE$est.int[1] + intHTE$est.int[2] * X_test + intHTE$est.int[3] * X_test^2
  
  print(intHTE$est.int)
  
  # Compute metrics
  rmse_values[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values[i] <- mean(true_tau - pred_tau)
  variance_values[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  bias_shuYang_df[[paste0("Iteration_", i)]] <- true_tau - pred_tau
}

# -------------------------------
# Visualize results
# -------------------------------
boxplot(rmse_values, main = "Boxplot of RMSE Values", ylab = "RMSE")
boxplot(bias_values, main = "Boxplot of Bias Values", ylab = "Bias")
boxplot(variance_values, main = "Boxplot of Variance Values", ylab = "Variance")

# -------------------------------
# Export results
# -------------------------------
results_shuYang_sim1 <- data.frame(
  "RMSE" = rmse_values,
  "Bias" = bias_values,
  "Variance" = variance_values
)

# Replace the paths below with your local folder
write.csv(results_shuYang_sim1, file = "/path/to/your/CausalICM/Simulation1/results_ShuYang.csv", row.names = FALSE)
write.csv(bias_shuYang_df, file = "/path/to/your/CausalICM/Simulation1/bias_ShuYang.csv", row.names = FALSE)
