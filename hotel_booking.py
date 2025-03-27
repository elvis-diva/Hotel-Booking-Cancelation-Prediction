import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("hotel_bookings.csv")

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

# Select features for training
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'babies', 'previous_cancellations',
            'previous_bookings_not_canceled', 'booking_changes',
            'deposit_type', 'customer_type']

# Target variable
target = 'is_canceled'

# Extract feature columns
X = df[features]

# Convert categorical columns into numerical format
X = pd.get_dummies(X, drop_first=True)

# Extract target column
y = df[target]

# Now `X` and `y` are defined, so we can split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot cancellation distribution
sns.countplot(x='is_canceled', data=df)
plt.title("Booking Cancellation Distribution")
plt.show()
