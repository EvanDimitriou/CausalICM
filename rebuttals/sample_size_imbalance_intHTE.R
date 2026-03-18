# -------------------------------
# Load datasets
# -------------------------------
datasets_sim <- read.csv("/Users/evangelosdimitriou/myGitRepo/CausalICM/CLeaR/rebuttals/diff_sample_size_data.csv")  # Use a relative or generic path

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
num_simulations <- 50
rmse_values_200 <- numeric(num_simulations)
bias_values_200 <- numeric(num_simulations)
variance_values_200 <- numeric(num_simulations)

# -------------------------------
# Loop over simulations (assume linear + quadratic effects)
# -------------------------------

# observational sample size = 200
for (i in 1:num_simulations) {
  cat("Simulation", i, "\n")
  
  data_rct <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==1 & datasets_sim$obs_sample_size==200, ]
  data_obs <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==0 & datasets_sim$obs_sample_size==200, ]
  
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
  rmse_values_200[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values_200[i] <- mean(true_tau - pred_tau)
  variance_values_200[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  
}

# -------------------------------
# Initialize storage vectors
# -------------------------------
num_simulations <- 50
rmse_values_500 <- numeric(num_simulations)
bias_values_500 <- numeric(num_simulations)
variance_values_500 <- numeric(num_simulations)

# -------------------------------
# Loop over simulations (assume linear + quadratic effects)
# -------------------------------

# observational sample size = 200
for (i in 1:num_simulations) {
  cat("Simulation", i, "\n")
  
  data_rct <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==1 & datasets_sim$obs_sample_size==500, ]
  data_obs <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==0 & datasets_sim$obs_sample_size==500, ]
  
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
  rmse_values_500[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values_500[i] <- mean(true_tau - pred_tau)
  variance_values_500[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  
}


# -------------------------------
# Initialize storage vectors
# -------------------------------
num_simulations <- 50
rmse_values_1000 <- numeric(num_simulations)
bias_values_1000 <- numeric(num_simulations)
variance_values_1000 <- numeric(num_simulations)

# -------------------------------
# Loop over simulations (assume linear + quadratic effects)
# -------------------------------

# observational sample size = 200
for (i in 1:num_simulations) {
  cat("Simulation", i, "\n")
  
  data_rct <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==1 & datasets_sim$obs_sample_size==1000, ]
  data_obs <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==0 & datasets_sim$obs_sample_size==1000, ]
  
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
  rmse_values_1000[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values_1000[i] <- mean(true_tau - pred_tau)
  variance_values_1000[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  
}


# -------------------------------
# Initialize storage vectors
# -------------------------------
num_simulations <- 50
rmse_values_2000 <- numeric(num_simulations)
bias_values_2000 <- numeric(num_simulations)
variance_values_2000 <- numeric(num_simulations)

# -------------------------------
# Loop over simulations (assume linear + quadratic effects)
# -------------------------------

# observational sample size = 200
for (i in 1:num_simulations) {
  cat("Simulation", i, "\n")
  
  data_rct <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==1 & datasets_sim$obs_sample_size==2000, ]
  data_obs <- datasets_sim[datasets_sim$datasetNo==i & datasets_sim$S==0 & datasets_sim$obs_sample_size==2000, ]
  
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
  rmse_values_2000[i] <- sqrt(mean((true_tau - pred_tau)^2))
  bias_values_2000[i] <- mean(true_tau - pred_tau)
  variance_values_2000[i] <- mean((pred_tau - mean(pred_tau))^2)
  
  
}




results_intHTE <- rbind(
  data.frame(
    obs_sample_size = 200,
    rep = 1:num_simulations,
    rmse = rmse_values_200,
    bias = bias_values_200,
    variance = variance_values_200
  ),
  data.frame(
    obs_sample_size = 500,
    rep = 1:num_simulations,
    rmse = rmse_values_500,
    bias = bias_values_500,
    variance = variance_values_500
  ),
  data.frame(
    obs_sample_size = 1000,
    rep = 1:num_simulations,
    rmse = rmse_values_1000,
    bias = bias_values_1000,
    variance = variance_values_1000
  ),
  data.frame(
    obs_sample_size = 2000,
    rep = 1:num_simulations,
    rmse = rmse_values_2000,
    bias = bias_values_2000,
    variance = variance_values_2000
  )
)

mean_rmse_200  <- mean(rmse_values_200)
mean_rmse_500  <- mean(rmse_values_500)
mean_rmse_1000 <- mean(rmse_values_1000)
mean_rmse_2000 <- mean(rmse_values_2000)
mean_rmse_200
mean_rmse_500
mean_rmse_1000
mean_rmse_2000

head(results_intHTE)
write.csv(results_intHTE,
          "/Users/evangelosdimitriou/myGitRepo/CausalICM/CLeaR/rebuttals/sample_size_imbalance_intHTE_results.csv",
          row.names = FALSE)
