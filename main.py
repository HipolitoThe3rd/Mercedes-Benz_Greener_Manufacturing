import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb

# 1. Load the Data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Separate target variable 'y' and identifiers 'ID'
y_train = train_df['y']
train_features = train_df.drop(['y', 'ID'], axis=1)
test_features = test_df.drop(['ID'], axis=1)

# ==============================================================================
# Action 1 & 2: Check for null and unique values for test and train sets
# ==============================================================================
print("\n--- Checking for Null Values ---")
print(f"Total null values in train set: {train_features.isnull().sum().sum()}")
print(f"Total null values in test set: {test_features.isnull().sum().sum()}")

print("\n--- Checking Unique Values ---")
# Checking the number of unique values for each categorical column
cat_cols = train_features.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"Unique values in {col} (Train): {train_features[col].nunique()} | (Test): {test_features[col].nunique()}")

# ==============================================================================
# Action 3: If for any column(s), the variance is equal to zero, remove them
# ==============================================================================
# Zero variance means the column has the exact same value for all rows.
# We calculate variance for numeric columns only.
numeric_cols = train_features.select_dtypes(include=[np.number]).columns
variances = train_features[numeric_cols].var()

zero_var_cols = variances[variances == 0].index.tolist()
print(f"\n--- Removing Zero Variance Columns ---")
print(f"Found {len(zero_var_cols)} columns with zero variance: {zero_var_cols}")

train_features.drop(columns=zero_var_cols, inplace=True)
test_features.drop(columns=zero_var_cols, inplace=True)

# ==============================================================================
# Action 4: Apply label encoder
# ==============================================================================
print("\n--- Applying Label Encoder ---")
# We apply label encoding to transform categorical (object) variables into numerical ones
for col in cat_cols:
    le = LabelEncoder()
    # We fit on the combined data of train and test to ensure all unique labels 
    # from both datasets are recognized and properly encoded without errors.
    combined_data = list(train_features[col].values) + list(test_features[col].values)
    le.fit(combined_data)
    
    train_features[col] = le.transform(train_features[col])
    test_features[col] = le.transform(test_features[col])

print("Label Encoding complete.")

# ==============================================================================
# Action 5: Perform dimensionality reduction (PCA)
# ==============================================================================
print("\n--- Performing Dimensionality Reduction (PCA) ---")
# We use Principal Component Analysis (PCA) to reduce the hundreds of features 
# into a smaller set of principal components that capture the most variance.
# 12 is a common starting component number for this specific dataset to prevent overfitting.
n_components = 12 
pca = PCA(n_components=n_components, random_state=42)

train_pca = pca.fit_transform(train_features)
test_pca = pca.transform(test_features)

print(f"Shape after PCA: Train {train_pca.shape}, Test {test_pca.shape}")

# ==============================================================================
# Action 6: Predict your test_df values using xgboost
# ==============================================================================
print("\n--- Training XGBoost and Predicting ---")
# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    learning_rate=0.1,
    max_depth=4,
    n_estimators=100,
    random_state=42
)

# Fit the model on the PCA-reduced training data
xgb_model.fit(train_pca, y_train)

# Predict the times for the test dataset
test_predictions = xgb_model.predict(test_pca)

# Create a final submission dataframe
submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'y': test_predictions
})

print("\n--- First 5 Predictions ---")
print(submission_df.head())

#Save predictions to a CSV file for submission
submission_df.to_csv('mercedes_xgboost_submission.csv', index=False)
print("\nPredictions saved to 'mercedes_xgboost_submission.csv'")