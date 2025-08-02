# # Untitled-1.py
# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 📥 2. Load Data
df = pd.read_csv("house_prices.csv")  # Replace with your actual filename
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 🔍 3. Initial Exploration
print(df.head())
print(df.info())
print(df.describe())

# 🧹 4. Clean & Preprocess Data
# Drop columns not useful for prediction
drop_cols = ['Index', 'Title', 'Description', 'Society', 'Dimensions']  # Not predictive or hard to quantify
df.drop(columns=drop_cols, inplace=True)

# Convert 'Amount(in rupees)' or 'Price (in rupees)' to numeric target
df.rename(columns={"Price (in rupees)": "Price"}, inplace=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)  # for categorical

# Convert area units to numeric (if needed)
def extract_number(x):
    try:
        return float(str(x).split()[0].replace(',', ''))
    except:
        return np.nan

df["Carpet Area"] = df["Carpet Area"].apply(extract_number)
df["Super Area"] = df["Super Area"].apply(extract_number)
df["Plot Area"] = df["Plot Area"].apply(extract_number)

# Label Encoding for categorical variables
categorical_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 🧪 5. Feature and Target Split
X = df.drop("Price", axis=1)
y = df["Price"]

# 🧱 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 7. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📈 8. Evaluate Model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("RMSE:", rmse)

# 📊 9. Feature Importance (Optional)
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6), title="Feature Importance")
plt.tight_layout()
plt.show()

# 💾 10. Save predictions (Optional)
pred_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
pred_df.to_csv('house_price_predictions.csv', index=False)
