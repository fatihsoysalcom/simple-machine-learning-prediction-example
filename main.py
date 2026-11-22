import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Generate synthetic data for demonstration
# We create a dataset where 'age' and 'income' influence 'buys_product'
np.random.seed(42)
num_samples = 100

# Features: Age (20-70), Income (30k-120k)
age = np.random.randint(20, 70, num_samples)
income = np.random.randint(30000, 120000, num_samples)

# Target: Buys product (0 or 1)
# Simple rule: Higher age AND higher income -> more likely to buy
# Add some noise to make it less perfectly linear and more realistic
buys_product = ((age > 45) & (income > 70000)).astype(int)
random_errors = np.random.rand(num_samples) < 0.2 # 20% chance to flip the outcome
buys_product = np.where(random_errors, 1 - buys_product, buys_product)

X = np.column_stack((age, income)) # Combine features into a single array
y = buys_product # Our target variable

print("--- Synthetic Data Sample ---")
for i in range(5):
    print(f"Age: {X[i, 0]}, Income: {X[i, 1]}, Buys Product: {y[i]}")
print("-" * 30)

# 2. Split data into training and testing sets
# This simulates having historical data to learn from (training) 
# and new, unseen data to predict on (testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training data size: {len(X_train)} samples")
print(f"Testing data size: {len(X_test)} samples")
print("-" * 30)

# 3. Choose and train a Machine Learning model
# Logistic Regression is a simple classification algorithm often used for binary outcomes.
# The model 'learns' the relationship between features (age, income) and the target (buys_product)
# by finding patterns in the training data.
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train) # This is the 'learning' phase of Machine Learning
print("Machine Learning model trained successfully!")
print("-" * 30)

# 4. Make predictions on the test set
# The trained model uses what it learned to predict outcomes for unseen data.
y_pred = model.predict(X_test)

# 5. Evaluate the model's performance
# We check how well the model's predictions match the actual outcomes in the test set.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on test data: {accuracy:.2f}")
print("-" * 30)

# 6. Demonstrate prediction for new, unseen individuals
print("--- Predicting for new individuals ---")
new_customers = np.array([
    [30, 40000],  # Young, low income
    [55, 90000],  # Older, high income
    [40, 60000],  # Middle-aged, middle income
    [60, 50000]   # Older, lower income
])

# The model predicts whether these new customers will buy the product
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers) # Get probabilities for each class

for i, customer in enumerate(new_customers):
    print(f"Customer {i+1} (Age: {customer[0]}, Income: {customer[1]}):")
    print(f"  Predicted to buy product: {'Yes' if predictions[i] == 1 else 'No'}")
    print(f"  Probability of buying: {probabilities[i, 1]:.2f}") # Probability of class 1 (buying)
print("-" * 30)

print("\nExplanation:")
print("This example demonstrates the core concept of Machine Learning: learning from data.")
print("1. We created synthetic data representing customer attributes and purchase behavior.")
print("2. This data was split into a training set (for the model to learn from) and a testing set (to evaluate its performance).")
print("3. A Logistic Regression model was 'trained' on the training data. This is where the machine 'learns' patterns and relationships.")
print("4. Finally, the trained model was used to predict outcomes for new, unseen customers.")
print("This process allows computers to make intelligent decisions or predictions based on learned patterns, without being explicitly programmed with every possible rule.")
