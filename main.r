library(ggplot2)
library(dplyr)
library(corrplot)
library(caret)
library(naniar)
library(GGally)
library(lubridate)
library(reshape2)
library(ggcorrplot)
library(Amelia)
library(gridExtra)
library(grid)
library(e1071)
library(MASS)
library(forecast)
library(moments)
library(VIM)
library(doParallel)
library(kernlab)
library(glmnet)
library(earth)

# Load the dataset
data <- read.csv("D:/Predictive/project 3/fifa_eda_stats.csv", stringsAsFactors = FALSE)

str(data)
summary(data)

# === Data Exploration & Visualization ===

categorical_vars <- names(data)[sapply(data, is.character)]
continuous_vars <- names(data)[sapply(data, is.numeric)]

variable_counts <- data.frame(
  Type = c("Categorical", "Continuous"),
  Count = c(length(categorical_vars), length(continuous_vars))
)

ggplot(variable_counts, aes(x = Type, y = Count, fill = Type)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Count), vjust = -0.5) +
  theme_minimal() +
  labs(title = "Count of Categorical and Continuous Variables", x = "Variable Type", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5)) + scale_fill_manual(values = c("Categorical" = "#e2f396", "Continuous" = "#02585a"))

for (var in continuous_vars) {
  p <- ggplot(data, aes(y = .data[[var]])) +
    geom_boxplot(fill = "#34a65e", color = "black") +
    theme_minimal() +
    labs(title = paste("Boxplot of", var), x = NULL, y = NULL) +
    theme(plot.title = element_text(hjust = 0.5, size = 20))
  print(p)
}

for (var in categorical_vars) {
  p <- ggplot(data, aes(x = .data[[var]])) +
    geom_bar(fill = "#34a65e", color = "black") +
    theme_minimal() +
    labs(title = var, x = NULL, y = "Count") + 
    theme(plot.title = element_text(hjust = 0.5, size = 20))
  print(p)
}

# === Transform Columns ===

# Convert Currency to Numeric Values
currency_convert <- function(x) {
  x <- gsub('€', '', x)
  if (grepl('M', x)) {
    as.numeric(gsub('M', '', x)) * 10^6
  } else if (grepl('K', x)) {
    as.numeric(gsub('K', '', x)) * 10^3
  } else {
    as.numeric(x)
  }
}

data$Value <- sapply(data$Value, currency_convert)
data$Wage <- sapply(data$Wage, currency_convert)
data$Release.Clause <- sapply(data$Release.Clause, currency_convert)

# Convert Height to cm and Weight to kg
convert_height <- function(height) {
  parts <- strsplit(height, "'")[[1]]
  feet <- as.numeric(parts[1])
  inches <- as.numeric(parts[2])
  cm <- (feet * 30.48) + (inches * 2.54)
  return(cm)
}

data$Height <- sapply(data$Height, convert_height)
data$Weight <- as.numeric(gsub("lbs", "", data$Weight))
head(data[, c("Height", "Weight", "Value", "Wage")])

str(data)

# === Check for Missing Values and Remove missing values  ===

check_missing_values <- function(data) {
  missing_data <- sapply(data, function(x) {
    sum(is.na(x) | x == "" | x == "None" | x == "NULL" | x == "Not Available" | x == "NaN")
  })
  return(missing_data)
}

missing_values <- check_missing_values(data)
cat("Missing values per column:\n")
print(missing_values)
total_missing <- sum(missing_values) / (nrow(data) * ncol(data)) * 100
total_complete <- 100 - total_missing
missmap(data, 
        main = "Missing Values", 
        col = c("white", "#02585a"),
        legend = FALSE)      

legend("topright", legend = c(
  paste("Missing Data: ", round(total_missing, 2), "%"),
  paste("Complete Data: ", round(total_complete, 2), "%")),
  bty = "n",
  text.col = "white",
  cex = 0.8)

# === Remove Unwanted Columns ===

data_cleaned <- data[, !(names(data) %in% c("ID", "Name", "Nationality", "Club", "Joined", "Contract.Valid.Until", "Loaned.From"))]
str(data_cleaned)

# === Handle Missing Values ===

missing_values2 <- check_missing_values(data_cleaned)
cat("Missing values per column:\n")
print(missing_values2)

categorical_columns <- names(data_cleaned)[sapply(data_cleaned, function(x) is.character(x) | is.factor(x))]
numerical_columns <- names(data_cleaned)[sapply(data_cleaned, is.numeric)]

cat("Categorical columns:\n")
print(categorical_columns)
cat("Numerical columns:\n")
print(numerical_columns)

# Impute Missing Values for Numerical Variables Using KNN
numerical_columns <- numerical_columns[numerical_columns %in% colnames(data_cleaned)]
data_cleaned[numerical_columns] <- lapply(data_cleaned[numerical_columns], as.numeric)

if (length(numerical_columns) > 0) {
  cat("Applying KNN imputation to numerical columns:\n")
  print(numerical_columns)
  data_cleaned <- kNN(data_cleaned, variable = numerical_columns, k = 5)

  data_cleaned <- data_cleaned[, !grepl("imp$", names(data_cleaned))]
} else {
  cat("No numerical columns found for KNN imputation.\n")
}

# Impute Missing Values for Categorical Variables Using Mode
mode_impute <- function(x) {
  uniq <- unique(na.omit(x))
  uniq[which.max(tabulate(match(x, uniq)))]
}

is_missing_like <- function(x) {
  return(is.na(x) | x == "" | tolower(x) %in% c("none", "null", "not available", "nan"))
}

if (length(categorical_columns) > 0) {
  for (col in categorical_columns) {
    # Check if there are any missing-like values (NA, "", "None", "NULL", "Not Available", "NaN")
    missing_values <- is_missing_like(data_cleaned[[col]])
    
    if (any(missing_values)) {
      # Impute missing-like values with the mode
      data_cleaned[[col]][missing_values] <- mode_impute(data_cleaned[[col]])
    }
  }
} else {
  cat("No categorical columns found for mode imputation.\n")
}

cat("Checking for remaining missing values after imputation:\n")
remaining_missing <- check_missing_values(data_cleaned)
print(remaining_missing)

total_missing <- sum(remaining_missing)
cat("Total missing values after imputation: ", total_missing, "\n")
cat("Sample size: ", nrow(data_cleaned), "\n")
cat("Number of predictors: ", ncol(data_cleaned)-1, "\n")

# === Data Transformations for Continuous Variables ===

continuous_vars_cleaned <- setdiff(names(data_cleaned)[sapply(data_cleaned, is.numeric)], "Wage")
continuous_data_cleaned <- data_cleaned[, continuous_vars_cleaned]

corr_matrix_cleaned <- cor(continuous_data_cleaned, use = "complete.obs")
ggcorrplot(corr_matrix_cleaned, 
           hc.order = TRUE,                
           type = "full",                  
           lab = FALSE,                    
           outline.color = "white",        
           colors = c("#e2f396", "white", "#02585a"),  
           tl.cex = 4,                     
           ggtheme = theme_minimal(),      
           title = "Correlation Matrix of Predictors") + 
  theme(legend.position = "right",    
        legend.key.height = unit(1.8, 'cm'), 
        legend.key.width = unit(0.3, 'cm'),
        plot.title = element_blank(),  
        legend.title = element_text(size = 0))  

# === Skewness Calculation Before and After Transformation ===

# Filter continuous variables for skewness calculation
numerical_predictors <- data_cleaned[, continuous_vars_cleaned]

skewValues_before <- apply(numerical_predictors, 2, skewness)
skewness_df_before <- data.frame(
  Predictor = names(skewValues_before),
  Skewness_Before = skewValues_before,
  Interpretation_Before = ifelse(skewValues_before > 1, "Heavily right skewed",
                                 ifelse(skewValues_before < -1, "Heavily left skewed",
                                        ifelse(skewValues_before > -1/2 & skewValues_before <= 1/2, "Approximately symmetric",
                                               ifelse(skewValues_before > 1/2 & skewValues_before <= 1, "Moderately right skewed",
                                                      ifelse(skewValues_before < -1/2 & skewValues_before >= -1, "Moderately left skewed", "Symmetric")
                                               )
                                        )
                                 )
  )
)

print(skewness_df_before)

# === Apply Box-Cox Transformation, Centering, and Scaling ===

trans <- preProcess(numerical_predictors, method = c("BoxCox", "center", "scale"))
transformed_data <- predict(trans, numerical_predictors)

skewValues_after <- apply(transformed_data, 2, skewness)

skewness_df_after <- data.frame(
  Skewness_After = skewValues_after,
  Interpretation_After = ifelse(skewValues_after > 1, "Heavily right skewed",
                                ifelse(skewValues_after < -1, "Heavily left skewed",
                                       ifelse(skewValues_after > -1/2 & skewValues_after <= 1/2, "Approximately symmetric",
                                              ifelse(skewValues_after > 1/2 & skewValues_after <= 1, "Moderately right skewed",
                                                     ifelse(skewValues_after < -1/2 & skewValues_after >= -1, "Moderately left skewed", "Symmetric")
                                              )
                                       )
                                )
  )
)

print(skewness_df_after)

plot_density_before <- function(df, var) {
  ggplot(df, aes(x = !!sym(var))) +  
    geom_density(fill = "#18b5d9", alpha = 0.5) +
    ggtitle(paste(var, "-Before Transformation")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 25))
}

plot_density_after <- function(df, var) {
  ggplot(df, aes(x = !!sym(var))) +  
    geom_density(fill = "#67bb0f", alpha = 0.5) +
    ggtitle(paste(var, "-After Transformation")) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 25))
}

for (var in continuous_vars_cleaned) {
  print(plot_density_before(numerical_predictors, var))
  print(plot_density_after(transformed_data, var))
}

comparison_df <- cbind(skewness_df_before, skewness_df_after)

SST <- preProcess(numerical_predictors, method = c("center", "scale"))
transformed_spatial_sign <- predict(SST, numerical_predictors)
spatial_sign_data <- spatialSign(transformed_spatial_sign)

par(mar = c(10, 4, 4, 2) + 0.1)  
boxplot(numerical_predictors,
        las = 2,  
        col = "#e2f396",  
        border = "#485702",  
        main = "Boxplot of Original Continuous Variables", 
        cex.axis = 0.8,  
        cex.main = 1.5,  
        cex.lab = 1.2,   
        notch = TRUE,  
        outline = FALSE)  
grid(nx = NA, ny = NULL, lty = "dotted", col = "gray")

boxplot(spatial_sign_data,
        las = 2,  
        col = "#289c83",  
        border = "#02585a",  
        main = "Boxplot of Spatial Sign Transformed Data", 
        cex.axis = 0.8,  
        cex.main = 1.5,  
        cex.lab = 1.2,   
        notch = TRUE,  
        outline = FALSE)  
grid(nx = NA, ny = NULL, lty = "dotted", col = "gray")

# === Create Dummy Variables for Categorical Columns ===

dummies_model <- dummyVars(~ Preferred.Foot + Work.Rate + Body.Type + Position, 
                           data = data_cleaned, 
                           fullRank = TRUE)

dummy_data_selected <- predict(dummies_model, newdata = data_cleaned)
dummy_vars <- colnames(dummy_data_selected)
combined_data <- cbind(continuous_data_cleaned, dummy_data_selected)
cat("Sample size: ", nrow(combined_data), "\n")  
cat("Number of predictors: ", ncol(combined_data), "\n")

str(combined_data)

# ===  Correlation and Non-Zero variance filtering  ===

cormat <- cor(combined_data, use = "pairwise.complete.obs")
dim(cormat)

ggcorrplot(cormat, 
           hc.order = FALSE,                
           type = "full",                  
           lab = FALSE,                    
           outline.color = "white",        
           colors = c("#e2f396", "white", "#02585a"),  
           tl.cex = 4,                     
           ggtheme = theme_minimal(),      
           title = "Correlation Matrix of Predictors") + 
  theme(legend.position = "right",    
        legend.key.height = unit(1.8, 'cm'), 
        legend.key.width = unit(0.3, 'cm'),
        plot.title = element_blank(),  
        legend.title = element_text(size = 0))

correlations <- cor(combined_data)
highCorr <- findCorrelation(correlations, cutoff = 0.90)
filtered_data_corr <- combined_data[, -highCorr]

corr_filtered <- cor(filtered_data_corr)
ggcorrplot(corr_filtered, 
           hc.order = FALSE,                
           type = "full",                  
           lab = FALSE,                    
           outline.color = "white",        
           colors = c("#e2f396", "white", "#02585a"),  
           tl.cex = 4,                     
           ggtheme = theme_minimal(),      
           title = "Correlation Matrix of Predictors") + 
  theme(legend.position = "right",    
        legend.key.height = unit(1.8, 'cm'), 
        legend.key.width = unit(0.3, 'cm'),
        plot.title = element_blank(),  
        legend.title = element_text(size = 0))

# Remove Near-Zero Variance Features
nzv <- nearZeroVar(filtered_data_corr, saveMetrics = TRUE)
nzv_features_corr <- rownames(nzv[nzv$nzv == TRUE, ])
filtered_data_final <- filtered_data_corr[, -which(names(filtered_data_corr) %in% nzv_features_corr)]

# Apply Near-Zero Variance Filtering to Original Dataset (without correlation filtering)
nzv_original <- nearZeroVar(combined_data, saveMetrics = TRUE)
nzv_features_original <- rownames(nzv_original[nzv_original$nzv == TRUE, ])
filtered_data_no_corr <- combined_data[, -which(names(combined_data) %in% nzv_features_original)]

cat("Structure of final dataset after removing correlation and near-zero variance features:\n")
str(filtered_data_final)

cat("Structure of final dataset after removing only near-zero variance features:\n")
str(filtered_data_no_corr)

registerDoParallel(cores = detectCores())
ctrl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)
trainIndex_final <- createDataPartition(data_cleaned$Wage, p = 0.8, list = FALSE)
trainData_final <- data_cleaned[trainIndex_final, ]
testData_final <- data_cleaned[-trainIndex_final, ]

trainIndex_no_corr <- createDataPartition(data_cleaned$Wage, p = 0.8, list = FALSE)
trainData_no_corr <- data_cleaned[trainIndex_no_corr, ]
testData_no_corr <- data_cleaned[-trainIndex_no_corr, ]

X_final_train <- filtered_data_final[trainIndex_final, ]
X_final_test <- filtered_data_final[-trainIndex_final, ]

X_no_corr_train <- filtered_data_no_corr[trainIndex_no_corr, ]
X_no_corr_test <- filtered_data_no_corr[-trainIndex_no_corr, ]

y_train <- trainData_final$Wage
y_test <- testData_final$Wage

cat("\nTrain Set Dimensions (filtered_data_final): ", dim(X_final_train), "\n")
cat("Test Set Dimensions (filtered_data_final): ", dim(X_final_test), "\n")

cat("\nTrain Set Dimensions (filtered_data_no_corr): ", dim(X_no_corr_train), "\n")
cat("Test Set Dimensions (filtered_data_no_corr): ", dim(X_no_corr_test), "\n")


# --- Linear Regression Model ---
linear_model <- train(x = X_final_train, y = y_train, method = "lm", trControl = ctrl, preProcess = c("center", "scale"))
linear_predictions <- predict(linear_model, newdata = X_final_test)
mse_linear <- mean((linear_predictions - y_test)^2)
rmse_linear <- sqrt(mse_linear)
r2_linear <- cor(y_test, linear_predictions)^2
print(linear_model)
cat("\nLinear Regression Results: RMSE =", rmse_linear, ", R2 =", r2_linear, "\n")
plot(linear_model)

# --- Partial Least Squares (PLS) Model ---
pls_grid <- expand.grid(.ncomp = 1:10)
pls_model <- train(x = X_no_corr_train, y = y_train, method = "pls", tuneGrid = pls_grid, trControl = ctrl, preProcess = c("center", "scale"))
pls_predictions <- predict(pls_model, newdata = X_no_corr_test)
mse_pls <- mean((pls_predictions - y_test)^2)
rmse_pls <- sqrt(mse_pls)
r2_pls <- cor(y_test, pls_predictions)^2
print(pls_model)
cat("\nPLS Model Results: RMSE =", rmse_pls, ", R2 =", r2_pls, "\n")
plot(pls_model)

# --- Ridge Regression Model ---
ridge_grid <- expand.grid(.lambda = seq(0, 1, length = 15))
ridge_model <- train(x = X_no_corr_train, y = y_train, method = "ridge", tuneGrid = ridge_grid, trControl = ctrl, preProcess = c("center", "scale"))
ridge_predictions <- predict(ridge_model, newdata = X_no_corr_test)
mse_ridge <- mean((ridge_predictions - y_test)^2)
rmse_ridge <- sqrt(mse_ridge)
r2_ridge <- cor(y_test, ridge_predictions)^2
print(ridge_model)
cat("\nRidge Regression Results: RMSE =", rmse_ridge, ", R2 =", r2_ridge, "\n")
plot(ridge_model)

# --- Lasso Regression Model ---
lasso_grid <- expand.grid(.fraction = seq(0.01, 1, length = 20))
lasso_model <- train(x = X_no_corr_train, y = y_train, method = "lasso", tuneGrid = lasso_grid, trControl = ctrl, preProcess = c("center", "scale"))
lasso_predictions <- predict(lasso_model, newdata = X_no_corr_test)
mse_lasso <- mean((lasso_predictions - y_test)^2)
rmse_lasso <- sqrt(mse_lasso)
r2_lasso <- cor(y_test, lasso_predictions)^2
print(lasso_model)
cat("\nLasso Regression Results: RMSE =", rmse_lasso, ", R2 =", r2_lasso, "\n")
plot(lasso_model)

# --- Elastic Net Model ---
enet_grid <- expand.grid(.lambda = c(0, 0.01, .1), .fraction = seq(0.01, 1, length = 20))
enet_model <- train(x = X_no_corr_train, y = y_train, method = "enet", tuneGrid = enet_grid, trControl = ctrl, preProcess = c("center", "scale"))
enet_predictions <- predict(enet_model, newdata = X_no_corr_test)
mse_enet <- mean((enet_predictions - y_test)^2)
rmse_enet <- sqrt(mse_enet)
r2_enet <- cor(y_test, enet_predictions)^2
print(enet_model)
cat("\nElastic Net Results: RMSE =", rmse_enet, ", R2 =", r2_enet, "\n")
plot(enet_model)

# --- Summary of Model Results ---
linear_models_results <- data.frame(
  Model = c("Linear Regression", "PLS", "Ridge", "Lasso", "Elastic Net"),
  RMSE = c(rmse_linear, rmse_pls, rmse_ridge, rmse_lasso, rmse_enet),
  R2 = c(r2_linear, r2_pls, r2_ridge, r2_lasso, r2_enet)
)
print(linear_models_results)

ggplot(linear_models_results, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "RMSE for Linear Models", x = "Model", y = "RMSE") +
  theme_minimal()
ggplot(linear_models_results, aes(x = Model, y = R2)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "R² for Linear Models", x = "Model", y = "R²") +
  theme_minimal()

# --- Non-Linear Models ---

registerDoParallel(cores = detectCores())
ctrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# --- Neural Network (NN) ---
nnet_grid <- expand.grid( .decay = c(0, 0.01, 0.1), .size = c(1:10), .bag = FALSE)
nnet_model <- train(x = X_final_train, y = y_train, method = "avNNet", tuneGrid = nnet_grid, 
                    trControl = ctrl, preProc = c("center", "scale"), linout = TRUE, trace = FALSE, 
                    MaxNWts = 10 * (ncol(X_final_train) + 1) + 10 + 1, maxit = 300)
nnet_predictions <- predict(nnet_model, newdata = X_final_test)
mse_nnet <- mean((nnet_predictions - y_test)^2)
rmse_nnet <- sqrt(mse_nnet)
r2_nnet <- cor(y_test, nnet_predictions)^2
print(nnet_model)
cat("\nNeural Network Results: RMSE =", rmse_nnet, ", R2 =", r2_nnet, "\n")
plot(nnet_model)

# --- MARS Model ---
mars_grid <- expand.grid(.degree = 1:3, .nprune = 2:40)
mars_model <- train(x = X_final_train, y = y_train, method = "earth", tuneGrid = mars_grid, trControl = ctrl)
mars_predictions <- predict(mars_model, newdata = X_final_test)
mse_mars <- mean((mars_predictions - y_test)^2)
rmse_mars <- sqrt(mse_mars)
r2_mars <- cor(y_test, mars_predictions)^2
print(mars_model)
cat("\nMARS Model Results: RMSE =", rmse_mars, ", R2 =", r2_mars, "\n")
plot(mars_model)

# --- KNN Model ---
knn_model <- train(x = X_final_train, y = y_train, method = "knn", preProc = c("center", "scale"), 
                   tuneLength = 20, trControl = ctrl)
knn_predictions <- predict(knn_model, newdata = X_final_test)
mse_knn <- mean((knn_predictions - y_test)^2)
rmse_knn <- sqrt(mse_knn)
r2_knn <- cor(y_test, knn_predictions)^2
print(knn_model)
cat("\nKNN Model Results: RMSE =", rmse_knn, ", R2 =", r2_knn, "\n")
plot(knn_model)

# --- SVM Model ---
svm_model <- train(x = X_final_train, y = y_train, method = "svmRadial", preProc = c("center", "scale"), 
                   tuneLength = 20, trControl = ctrl)
svm_predictions <- predict(svm_model, newdata = X_final_test)
mse_svm <- mean((svm_predictions - y_test)^2)
rmse_svm <- sqrt(mse_svm)
r2_svm <- cor(y_test, svm_predictions)^2
print(svm_model)
cat("\nSVM Model Results: RMSE =", rmse_svm, ", R2 =", r2_svm, "\n")
ggplot(svm_model) + coord_trans(x = 'log2') + theme_bw()


# --- Non-Linear Models Results Summary ---
non_linear_models_results <- data.frame(
  Model = c("Neural Network", "MARS", "KNN", "SVM"),
  RMSE = c(rmse_nnet, rmse_mars, rmse_knn, rmse_svm),
  R2 = c(r2_nnet, r2_mars, r2_knn, r2_svm)
)

print(non_linear_models_results)

# --- Important Variables of MARS ---
imp <- varImp(mars_model)
print(imp)
plot(imp, top=10)

