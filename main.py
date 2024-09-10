import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import base64

# Step 1: Feature Extraction (no change)

def extract_features(ciphertext):
    """Extract features from ciphertext without adding non-numeric labels."""
    # Try decoding as hex
    try:
        byte_array = np.frombuffer(bytes.fromhex(ciphertext), dtype=np.uint8)
    except ValueError:
        # Try decoding as Base64
        missing_padding = len(ciphertext) % 4
        if missing_padding:
            ciphertext += '=' * (4 - missing_padding)
        try:
            byte_array = np.frombuffer(base64.b64decode(ciphertext), dtype=np.uint8)
        except Exception as e:
            print(f"Error decoding ciphertext: {e}")
            # Fall back to treating the ciphertext as raw bytes
            byte_array = np.frombuffer(ciphertext.encode(), dtype=np.uint8)
    
    # Ensure byte_array is valid and non-empty
    if len(byte_array) == 0 or byte_array.ndim != 1:
        print(f"Invalid or empty ciphertext: {ciphertext}")
        return np.zeros(256 + 3)  # Return a default feature vector of fixed length (without encoding label)

    # Length of ciphertext
    length = len(byte_array)

    # Calculate entropy
    byte_counts = np.bincount(byte_array, minlength=256)
    probabilities = byte_counts / len(byte_array)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))

    # Number of unique bytes
    unique_bytes = np.count_nonzero(byte_counts)

    # Append byte frequency distribution
    feature_vector = [length, entropy, unique_bytes] + byte_counts.tolist()

    return np.array(feature_vector)







# Step 2: Train the Machine Learning Model
def train_model_from_csv(csv_file):
    """Train a Random Forest classifier using a CSV file with columns 'algorithm' and 'cipher_text'."""
    
    # Read CSV
    data = pd.read_csv(csv_file)
    
    # Extract features from each ciphertext
    feature_matrix = np.array([extract_features(ciphertext) for ciphertext in data['cipher_text']])
    
    # Labels
    labels = data['algorithm'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate the model using cross-validation
    scores = cross_val_score(best_model, feature_matrix, labels, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean()}")

    # Save the trained model
    joblib.dump(best_model, 'crypto_classifier_model_from_csv.pkl')

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Step 3: Predict the Algorithm (no change)
def predict_algorithm(ciphertext):
    """Predict the encryption algorithm used for the given ciphertext."""
    # Load the trained model
    model = joblib.load('crypto_classifier_model_from_csv.pkl')

    # Extract features from the input ciphertext
    features = extract_features(ciphertext)

    # Predict the algorithm
    predicted_algo = model.predict([features])

    return predicted_algo[0]  # Return the predicted algorithm

# Example Usage
def main():
    # Step 1: Train the model using the CSV file
    csv_file = 'encrypted_output.csv'  # Path to your CSV file
    train_model_from_csv(csv_file)

    # Step 2: Take user input for ciphertext
    user_ciphertext = input("Enter the ciphertext: ")

    # Step 3: Predict the algorithm
    result = predict_algorithm(user_ciphertext)
    print(f"The predicted cryptographic algorithm is: {result}")

if __name__ == "__main__":
    main()
