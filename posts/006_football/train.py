import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
import mlflow.sklearn

# Load data into a pandas DataFrame
df = pd.read_csv("2022_2023_Football_Player_Stats.csv", sep=";", encoding="latin1")

# Filter and preprocess data
df = df[(df["MP"] >= 5)]
df["PosMod"] = df["Pos"].str[:2]

# Encode categorical variable PosMod
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Class"] = label_encoder.fit_transform(df["PosMod"])

# Define columns to scale
cols = ["Shots", "PasMedAtt", "Pas3rd", "Clr"]

# Split data into train and test sets
X = df[cols]
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run():
    
    mlflow.sklearn.log_model(scaler, "rf_scaler")
    
    # Train Random Forest classifier
    rf = RandomForestClassifier(random_state=10)
    rf.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = rf.predict(X_test_scaled)

    # Evaluate the model
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")

    # Log metrics to MLflow
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)

    # Feature importance
    feature_importances = rf.feature_importances_
    important_features = []
    for col, imp in zip(columns_to_scale, feature_importances):
            important_features.append(f"{col}: {imp}")
    mlflow.log_param("important_features", important_features)
    
    mlflow.sklearn.log_model(rf, "rf_model")
