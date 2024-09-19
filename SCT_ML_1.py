# house_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

# Data exploration and visualization
def explore_data(data):
    """Perform basic exploratory data analysis."""
    print("Dataset Summary:")
    print(data.describe())
    
    print("\nCorrelation Matrix:")
    correlation_matrix = data.corr()
    print(correlation_matrix)

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Plot pairwise relationships
    sns.pairplot(data)
    plt.title('Pairwise Relationships')
    plt.show()

# Data preprocessing
def preprocess_data(data):
    """Clean the dataset by handling missing values."""
    print("\nChecking for missing values:")
    print(data.isnull().sum())

    # Handle missing values (drop or fill)
    data.dropna(inplace=True)  # Optionally handle missing values

    return data

# Feature selection
def select_features(data):
    """Select features and target variable."""
    X = data[['square_footage', 'bedrooms', 'bathrooms']]
    y = data['price']
    return X, y

# Train and evaluate the model
def train_model(X_train, y_train):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Visualize the predictions
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
    plt.show()

# Main function
def main():
    """Main function to run the entire process."""
    # Load data
    data = load_data('house_prices.csv')
    
    # Explore data
    explore_data(data)

    # Preprocess data
    cleaned_data = preprocess_data(data)

    # Select features
    X, y = select_features(cleaned_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Print model coefficients
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print("\nModel Coefficients:")
    print(coefficients)

if _name_ == "_main_":
    main()