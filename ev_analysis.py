# 1. Import libraries FIRST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load your data NEXT
df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/EV Sales model/Electric Vehicle Sales by State in India.csv")

# 3. Data inspection and cleaning
print(df.head())
df['Year'] = df['Year'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
categorical_columns = ['Month_Name', 'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']
df[categorical_columns] = df[categorical_columns].astype("category")
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# 4. Visualize yearly sales trends
plt.figure(figsize=(6,4))
plt.title('Yearly Analysis of EV Sales in India')
sns.lineplot(x='Year', y='EV_Sales_Quantity', data=df, marker='o', color='b')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.show()

# 5. Feature engineering
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df_encoded = pd.get_dummies(df, columns=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], drop_first=True)
df_encoded.drop(['Date', 'Month_Name'], axis=1, inplace=True)

# 6. Train/test split and model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted EV Sales')
plt.show()

# 8. Feature importance (optional but useful)
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df['Month_Name'] is a categorical variable with months in the correct order
plt.figure(figsize=(8,5))
sns.lineplot(x='Month_Name', y='EV_Sales_Quantity', data=df, marker='o', color='r')
plt.title('Monthly Analysis of EV Sales in India')
plt.xlabel('Month')
plt.ylabel('EV Sales')
plt.tight_layout()
plt.savefig('monthly_ev_sales.png')  # Save the image for report
plt.show()


state_sales = df.groupby('State')['EV_Sales_Quantity'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(y=state_sales.index, x=state_sales.values, palette='viridis')
plt.title('State-wise Analysis of EV Sales')
plt.xlabel('EV Sales')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('statewise_ev_sales.png')
plt.show()


category_sales = df.groupby('Vehicle_Category')['EV_Sales_Quantity'].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=category_sales.index, y=category_sales.values, palette='mako')
plt.title('EV Sales by Vehicle Category')
plt.xlabel('Vehicle Category')
plt.ylabel('EV Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('vehicle_category_ev_sales.png')
plt.show()


type_sales = df.groupby('Vehicle_Type')['EV_Sales_Quantity'].sum().sort_values(ascending=False)
plt.figure(figsize=(10,4))
sns.barplot(x=type_sales.index, y=type_sales.values, palette='cubehelix')
plt.title('EV Sales by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('EV Sales')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('vehicle_type_ev_sales.png')
plt.show()
