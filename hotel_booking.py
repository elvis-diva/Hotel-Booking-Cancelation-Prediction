import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("hotel_bookings.csv")

# Define custom labels
label_mapping = {0: "Not Canceled", 1: "Canceled"}

# Plot with renamed labels
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='is_canceled', data=df)
plt.title("Booking Cancellation Distribution")

# Set tick positions and labels
ax.set_xticks([0, 1])  # Ensure ticks match the category values
ax.set_xticklabels(["Not Canceled", "Canceled"])  # Custom labels

plt.xlabel("Booking Status")
plt.ylabel("Count")
plt.show()
# Display basic info
print(df.info())
print(df.head())

# Check missing values
print(df.isnull().sum())

# Drop columns with too many missing values
df = df.drop(columns=['company', 'agent'])

# Fill missing numerical values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing categorical values with the most frequent value
df.fillna(df.mode().iloc[0], inplace=True)

# Additional features to include
extra_features = ['total_of_special_requests', 'required_car_parking_spaces',
                  'market_segment', 'distribution_channel',
                  'reserved_room_type', 'assigned_room_type']

# Updated feature list
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'babies', 'previous_cancellations',
            'previous_bookings_not_canceled', 'booking_changes',
            'deposit_type', 'customer_type'] + extra_features

# Extract updated feature columns
X = df[features]

# Convert categorical features into numerical
X = pd.get_dummies(X, drop_first=True)

# Extract target variable
y = df['is_canceled']

# Re-run the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Updated Model Accuracy: {accuracy:.2f}")

# Plot cancellation distribution
sns.countplot(x='is_canceled', data=df)
plt.title("Booking Cancellation Distribution")
plt.show()
