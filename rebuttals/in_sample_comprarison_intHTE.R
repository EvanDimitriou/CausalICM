# import datasets
datasets_sim8 <- read.csv("~/myGitRepo/multi-task-GPs-for-Treatment-Effect-Estimation-2/__pycache__/neurIPS/Simulation 8/datasets_sim8.csv")

# Integrative HTE

devtools::install_github("shuyang1987/IntegrativeHTEcf")


library(mgcv)
#> Loading required package: nlme
#> This is mgcv 1.8-26. For overview type 'help("mgcv-package")'.
library(MASS)
library(rootSolve)

X_test <- seq(-2.0, 0.0, by = 0.04)

# Initialize vector to store RMSE values for each iteration
rmse_values <- numeric(100)
bias_values <- numeric(100)
variance_values <- numeric(100)


# Run the loop 50 times
for (i in 1:100) {
  print(i)
  data_rct = datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==1, ]
  data_obs = datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==0, ]
  
  # RCT data: A (treatment), X (covariate), U (some variable, treated as covariate)
  A1 <- data_rct[, "A"]
  X1 <- data_rct[, "X"]
  
  # Observational data: A, X, U
  A2 <- data_obs[, "A"]
  X2 <- data_obs[, "X"]
  
  # Combine RCT and observational data
  A <- c(A1, A2)
  X <- c(X1, X2)
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))  # Participation indicator
  Y <- c(data_rct[, "Y"], data_obs[, "Y"])
  
  # Covariates matrices for HTE model
  X.hte <- cbind(X, X^2)
  X.cf <- cbind(X)
  
  X <- as.matrix(X)
  X.hte <- as.matrix(X.hte)
  X.cf <- as.matrix(X.cf)
  
  # Fit the Integrative HTE model
  intHTE <- IntegrativeHTEcf::IntHTEcf(A, X, X.hte, X.cf, Y, S, nboots = 50)
  
  
  # True treatment effect (CATE)
  true_tau <- 1 + X_test + X_test^2
  
  # Predicted treatment effect using estimated model coefficients
  pred_tau <- intHTE$est.int[1] + intHTE$est.int[2] * X_test + intHTE$est.int[3] * X_test^2
  
  print(c(intHTE$est.int[1], intHTE$est.int[2], intHTE$est.int[3], intHTE$est.int[4], intHTE$est.int[5]))
  
  # Compute RMSE and store it
  rmse <- sqrt(mean((true_tau - pred_tau)^2))
  bias <- mean(true_tau - pred_tau)
  variance <- mean((pred_tau - mean(pred_tau))^2)
  
  rmse_values[i] <- rmse
  bias_values[i] <- bias
  variance_values[i] <- variance
  
}



in_sample_intHTE_results <- data.frame("RMSE" = rmse_values, "Bias" = bias_values, "Variance" = variance_values)

# Export the results to the specified folder as a CSV file
write.csv(in_sample_intHTE_results, file = "/Users/evangelosdimitriou/myGitRepo/CausalICM/CLeaR/rebuttals/in_sample_intHTE_results.csv", row.names = FALSE)






