# import datasets
datasets_sim8 <- read.csv("~/myGitRepo/multi-task-GPs-for-Treatment-Effect-Estimation-2/__pycache__/neurIPS/Simulation 8/datasets_sim8.csv")

# Integrative HTE

devtools::install_github("shuyang1987/IntegrativeHTEcf")


library(mgcv)
#> Loading required package: nlme
#> This is mgcv 1.8-26. For overview type 'help("mgcv-package")'.
library(MASS)
library(rootSolve)

X_test <- seq(-2.0, 2.0, by = 0.04)

# Initialize vector to store RMSE values for each iteration
rmse_values <- numeric(100)
bias_values <- numeric(100)
variance_values <- numeric(100)


# Initialize vector to store runtime per iteration
runtime_values <- numeric(100)

# Loop over 100 datasets
for (i in 1:100) {
  print(paste("Dataset", i))
  
  data_rct = datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==1, ]
  data_obs = datasets_sim8[datasets_sim8$datasetNo==i & datasets_sim8$S==0, ]
  
  # RCT and observational covariates
  A1 <- data_rct[, "A"]
  X1 <- data_rct[, "X"]
  A2 <- data_obs[, "A"]
  X2 <- data_obs[, "X"]
  
  # Combine data
  A <- c(A1, A2)
  X <- c(X1, X2)
  S <- c(rep(1, nrow(data_rct)), rep(0, nrow(data_obs)))
  Y <- c(data_rct[, "Y"], data_obs[, "Y"])
  
  # Covariate matrices
  X.hte <- cbind(X, X^2)
  X.cf <- cbind(X)
  
  X <- as.matrix(X)
  X.hte <- as.matrix(X.hte)
  X.cf <- as.matrix(X.cf)
  
  # -----------------------------
  # Measure runtime
  # -----------------------------
  start_time <- proc.time()  # start timer
  
  intHTE <- IntegrativeHTEcf::IntHTEcf(A, X, X.hte, X.cf, Y, S, nboots = 50)
  
  runtime_values[i] <- (proc.time() - start_time)["elapsed"]  # store elapsed seconds
  
  # True treatment effect
  X_test <- seq(-2.0, 2.0, by = 0.04)
  true_tau <- 1 + X_test + X_test^2
  
  # Predicted CATE
  pred_tau <- intHTE$est.int[1] + intHTE$est.int[2] * X_test + intHTE$est.int[3] * X_test^2
}

mean(runtime_values)
sd(runtime_values)
median(runtime_values)
min(runtime_values)
max(runtime_values)

runtime_values

# Combine results into a data frame
results_df <- data.frame(
  datasetNo = 1:100,
  Runtime_sec = runtime_values
)

# View first few rows
head(results_df)

# Save raw results to CSV
write.csv(results_df, "~/myGitRepo/CausalICM/CLeaR/rebuttals/IntegrativeHTE_runtime_results.csv", row.names = FALSE)
