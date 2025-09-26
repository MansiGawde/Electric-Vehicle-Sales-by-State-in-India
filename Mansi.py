# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/EV Sales model/Electric Vehicle Sales by State in India.csv")

# 3. Data Inspection and Cleaning
df['Year'] = df['Year'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

categorical_columns = ['Month_Name', 'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']
df[categorical_columns] = df[categorical_columns].astype("category")

print(df.info())
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

# 4. Yearly Analysis of EV Sales
plt.figure(figsize=(6,4))
sns.lineplot(x='Year', y='EV_Sales_Quantity', data=df, marker='o', color='b')
plt.title('Yearly Analysis of EV Sales in India')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.tight_layout()
plt.savefig('yearly_ev_sales.png')
plt.close()

# 5. Monthly Analysis of EV Sales
plt.figure(figsize=(8,5))
sns.lineplot(x='Month_Name', y='EV_Sales_Quantity', data=df, marker='o', color='r')
plt.title('Monthly Analysis of EV Sales in India')
plt.xlabel('Month')
plt.ylabel('EV Sales')
plt.tight_layout()
plt.savefig('monthly_ev_sales.png')
plt.close()

# 6. State-wise Analysis of EV Sales
state_sales = df.groupby('State', observed=True)['EV_Sales_Quantity'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(y=state_sales.index, x=state_sales.values)
plt.title('State-wise Analysis of EV Sales')
plt.xlabel('EV Sales')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('statewise_ev_sales.png')
plt.close()

# 7. EV Sales by Vehicle Category
cat_sales = df.groupby('Vehicle_Category', observed=True)['EV_Sales_Quantity'].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=cat_sales.index, y=cat_sales.values)
plt.title('EV Sales by Vehicle Category')
plt.xlabel('Vehicle Category')
plt.ylabel('EV Sales')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('vehicle_category_ev_sales.png')
plt.close()

# 8. EV Sales by Vehicle Type
type_sales = df.groupby('Vehicle_Type', observed=True)['EV_Sales_Quantity'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,4))
sns.barplot(x=type_sales.index, y=type_sales.values)
plt.title('EV Sales by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('EV Sales')
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig('vehicle_type_ev_sales.png')
plt.close()

# 9. Feature Engineering for Modeling
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df_encoded = pd.get_dummies(df, columns=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], drop_first=True)
df_encoded.drop(['Date', 'Month_Name'], axis=1, inplace=True)

# 10. Train/Test Split and Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 11. Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# 12. Plot Actual vs Predicted EV Sales
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual EV Sales')
plt.ylabel('Predicted EV Sales')
plt.title('Actual vs Predicted EV Sales')
plt.tight_layout()
plt.savefig('actual_vs_predicted_ev_sales.png')
plt.close()

# 13. Feature Importance Plot (Top 20)
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)[:20]
plt.figure(figsize=(10,5))
feature_importance.plot(kind='bar')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance_top20.png')
plt.close()

print("All analyses complete. Plots saved as PNG files.")