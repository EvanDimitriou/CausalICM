df_rct <- read.csv("/path/to/your/CausalICM/rwd_rct_SY.csv")
df_rwd <- read.csv("/path/to/your/CausalICM/rwd_obs_SY.csv")
df_eval <- read.csv("/path/to/your/CausalICM/rwd_eval_SY.csv")

names(df_rct) <- c("X1", "X2", "X3", "X4", "X5", "X6", "A", "Y")
names(df_rwd) <- c("X1", "X2", "X3", "X4", "X5", "X6", "A", "Y")
names(df_eval) <- c("X1", "X2", "X3", "X4", "X5", "X6", "A", "Y")

#df_rct[, 1:6] <- scale(df_rct[, 1:6])
#df_rwd[, 1:6] <- scale(df_rwd[, 1:6])
#df_eval[, 1:6] <- scale(df_eval[, 1:6])



devtools::install_github("shuyang1987/IntegrativeHTEcf")


library(mgcv)
#> Loading required package: nlme
#> This is mgcv 1.8-26. For overview type 'help("mgcv-package")'.
library(MASS)
library(rootSolve)


# RCT data: A (treatment), X (covariate), U (some variable, treated as covariate)
A1 <- df_rct[, "A"]
X1 <- df_rct[, c("X1", "X2", "X3", "X4", "X5", "X6")]

# Observational data: A, X, U
A2 <- df_rwd[, "A"]
X2 <- df_rwd[, c("X1", "X2", "X3", "X4", "X5", "X6")]

# Combine RCT and observational data
A <- c(A1, A2)
X <- rbind(X1, X2)
S <- c(rep(1, nrow(df_rct)), rep(0, nrow(df_rwd)))  # Participation indicator
Y <- c(df_rct[, "Y"], df_rwd[, "Y"])

# Covariates matrices for HTE model
X.hte <- cbind(X)
X.cf <- cbind(X)

X <- as.matrix(X)
X.hte <- as.matrix(X.hte)
X.cf <- as.matrix(X.cf)

# Fit the Integrative HTE model
intHTE <- IntegrativeHTEcf::IntHTEcf(A, X, X.hte, X.cf, Y, S, nboots = 50)
print(intHTE$est.int)


# Compute cate
cate_est <- intHTE$est.int["psi0"] +
  intHTE$est.int["psi1"] * df_eval[["X1"]] +
  intHTE$est.int["psi2"] * df_eval[["X2"]] +
  intHTE$est.int["psi3"]* df_eval[["X3"]] +
  intHTE$est.int["psi4"] * df_eval[["X4"]] +
  intHTE$est.int["psi5"] * df_eval[["X5"]] +
  intHTE$est.int["psi6"] * df_eval[["X6"]] 

cate_est
trueCATE<- read.csv("/path/to/your/CausalICM/trueCATE.csv")

rmse_STAR_integrativeHTE<-sqrt(mean(cate_est - trueCATE[[1]])^2)



rmse_file <- "/path/to/your/CausalICM/rmse_STAR_integrativeHTE.rds"
write.csv(rmse_STAR_integrativeHTE, file = "/path/to/your/CausalICM/rmse_STAR_integrativeHTE.csv", row.names = FALSE)




