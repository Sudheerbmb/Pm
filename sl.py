import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('pm.csv')

# Select features and target variables
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

X = df[features]
y = df[failure_types]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multi-output Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the model to a pickle file
with open('rf_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(rf_classifier, model_file)

# Function to predict machine failure type
def predict_failure(air_temp, process_temp, rotational_speed, torque, tool_wear):
    input_data = pd.DataFrame([[air_temp, process_temp, rotational_speed, torque, tool_wear]],
                              columns=features)

    with open('rf_classifier_model.pkl', 'rb') as model_file:
        rf_classifier = pickle.load(model_file)
    
    predictions = rf_classifier.predict(input_data)

    failure_types = {
        'TWF': 'Total Wear Failure',
        'OSF': 'Overstrain Failure',
        'HDF': 'Heat Dissipation Failure',
        'PWF': 'Power Failure',
        'RNF': 'Random Failure'  # Keeping RNF as is, since it wasn't specified in the mapping
    }

    predicted_failures = [failure_types[ft] for ft, pred in zip(failure_types.keys(), predictions[0]) if pred == 1]

    if not predicted_failures:
        return "No failure predicted"
    else:
        return "Predicted failure type(s): " + ', '.join(predicted_failures)

# Test the function
print(predict_failure(300, 310, 1500, 40, 200))
print(predict_failure(298, 308, 1400, 30, 100))
print(predict_failure(300.8, 309.4, 1342, 62.4, 113))

