# -------------------------------
# Load datasets
# -------------------------------
datasets_sim8 <- read.csv("datasets_sim8.csv")  # Use a relative or generic path

# -------------------------------
# Install & load required packages
# -------------------------------
# devtools::install_github("shuyang1987/IntegrativeHTEcf") # Uncomment if needed
library(IntegrativeHTEcf)
library(mgcv)
library(MASS)
library(rootSolve)

# -------------------------------
# Define test grid
# -------------------------------
X_test <- seq(-2.0, 2.0, by = 0.04)

# -------------------------------
# Initialize storage vectors
# -------------------------------
num_simulations <- 100
rmse_values <- numeric(num_simulations)
bias_values <- numeric(num_simulations)
variance_values <- numeric(num_simulations)
bias_shuYang_df <- data.frame("X_test" = X_test)

# -------------------------------
# Loop over simulations (assume linear + quadratic effects)
# -------------------------------
for (i in 1:num_simulations) {
  cat("Simulation", i, "\n")
  
  data_rct <- datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==1, ]
  data_obs <- datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==0, ]
  
  # Combine RCT and observational data
  A <- c(data_rct$A, data_obs$A)
  X <- c(data_rct$X, data_obs$X)
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))  # Participation indicator
  Y <- c(data_rct$Y, data_obs$Y)
  
  # Covariate matrices for HTE model
  X.hte <- as.matrix(cbind(X, X^2))
  X.cf  <- as.matrix(cbind(X))
  
  # Fit Integrative HTE model
  intHTE <- IntegrativeHTEcf::IntHTEcf(A, as.matrix(X), X.hte, X.cf, Y, S, nboots = 50)
  
  # True treatment effect
  true_tau <- 1 + X_test + X_test^2
  
  # Predicted treatment effect
  pred_tau <- intHTE$est.int[1] + intHTE$est.int[2] * X_test + intHTE$est.int[3] * X_test^2
  
  # Compute metrics
  rmse_values[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values[i] <- mean(true_tau - pred_tau)
  variance_values[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  # Store bias for each iteration
  bias_shuYang_df[[paste0("Iteration_", i)]] <- true_tau - pred_tau
}

# -------------------------------
# Boxplots of metrics
# -------------------------------
boxplot(rmse_values, main = "Boxplot of RMSE Values", ylab = "RMSE")
boxplot(bias_values, main = "Boxplot of Bias Values", ylab = "Bias")
boxplot(variance_values, main = "Boxplot of Variance Values", ylab = "Variance")

# -------------------------------
# Save results
# -------------------------------
results_shuYang_sim8 <- data.frame("RMSE" = rmse_values, "Bias" = bias_values, "Variance" = variance_values)
write.csv(results_shuYang_sim8, file = "results_ShuYang.csv", row.names = FALSE)
write.csv(bias_shuYang_df, file = "bias_ShuYang.csv", row.names = FALSE)

# -------------------------------
# Quadratic covariates only version
# -------------------------------
rmse_values_quad <- numeric(num_simulations)
bias_values_quad <- numeric(num_simulations)
variance_values_quad <- numeric(num_simulations)

for (i in 1:num_simulations) {
  cat("Quadratic Simulation", i, "\n")
  
  data_rct <- datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==1, ]
  data_obs <- datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==0, ]
  
  A <- c(data_rct$A, data_obs$A)
  X <- c(data_rct$X, data_obs$X)
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))
  Y <- c(data_rct$Y, data_obs$Y)
  
  X.hte <- as.matrix(cbind(X, X^2))
  X.cf  <- as.matrix(cbind(X, X^2))
  
  intHTE <- IntegrativeHTEcf::IntHTEcf(A, as.matrix(X), X.hte, X.cf, Y, S, nboots = 50)
  
  true_tau <- 1 + X_test + X_test^2
  pred_tau <- intHTE$est.int[1] + intHTE$est.int[2] * X_test + intHTE$est.int[3] * X_test^2
  
  rmse_values_quad[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values_quad[i] <- mean(true_tau - pred_tau)
  variance_values_quad[i] <- mean((pred_tau - mean(pred_tau))^2)
}

boxplot(rmse_values_quad, main = "Boxplot of RMSE Values (Quadratic)", ylab = "RMSE")
boxplot(bias_values_quad, main = "Boxplot of Bias Values (Quadratic)", ylab = "Bias")
boxplot(variance_values_quad, main = "Boxplot of Variance Values (Quadratic)", ylab = "Variance")

results_shuYang_sim8_quad <- data.frame("RMSE" = rmse_values_quad, "Bias" = bias_values_quad, "Variance" = variance_values_quad)
write.csv(results_shuYang_sim8_quad, file = "results_ShuYang_quad.csv", row.names = FALSE)
