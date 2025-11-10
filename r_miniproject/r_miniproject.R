# Install required packages
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(randomForest)) install.packages("randomForest")
if (!require(smotefamily)) install.packages("smotefamily")

# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(smotefamily)

# Step 1: Load the dataset
credit_data <- read.csv("C:/Users/Mallika/Downloads/credit.csv")

# Step 2: Exploratory Data Analysis (EDA)
cat("Structure of the dataset:\n")
print(str(credit_data))
cat("Summary of the dataset:\n")
print(summary(credit_data))
cat("Class distribution:\n")
print(table(credit_data$Class))  # Check class imbalance

# Step 3: Data Preprocessing
credit_data$Class <- as.factor(credit_data$Class)  # Convert Class to a factor
set.seed(42)  # For reproducibility

# SMOTE: Handle class imbalance
smote_output <- SMOTE(X = credit_data[, -which(names(credit_data) == "Class")], 
                      target = credit_data$Class, 
                      K = 5, 
                      dup_size = 2)
smote_data <- smote_output$data
colnames(smote_data)[ncol(smote_data)] <- "Class"  # Rename the target column
smote_data$Class <- as.factor(smote_data$Class)    # Ensure Class is a factor

cat("Class distribution after SMOTE:\n")
print(table(smote_data$Class))  # Verify balanced classes

# Step 4: Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(smote_data$Class, p = 0.8, list = FALSE)
train_data <- smote_data[train_index, ]
test_data <- smote_data[-train_index, ]

# Step 5: Train the Random Forest Model
set.seed(42)
rf_model <- randomForest(Class ~ ., data = train_data, ntree = 100, importance = TRUE)

# Step 6: Evaluate feature importance
cat("Feature Importance:\n")
print(importance(rf_model))
varImpPlot(rf_model)  # Visualize feature importance

# Step 7: Make predictions on the test set
predictions <- predict(rf_model, test_data)

# Step 8: Evaluate the model
confusion_matrix <- confusionMatrix(predictions, test_data$Class)
cat("Confusion Matrix:\n")
print(confusion_matrix)