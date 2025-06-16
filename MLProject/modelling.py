import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

df = pd.read_csv("dataset_preprocessing/personality_dataset_preprocessing.csv")
X = df.drop("Personality", axis=1)
y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("model_type", "Logistic Regression with Scaler")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=infer_signature(X_train, model.predict(X_train)),
        input_example=X_train.head(5)
    )

    # Save and log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)

    # Simpan model & confusion matrix ke folder saved_models
    model_dir = f"saved_models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # Simpan model
    model_path = f"{model_dir}/model.pkl"
    joblib.dump(best_model, model_path)

    # Simpan confusion matrix
    cm_path = f"{model_dir}/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close() 

    mlflow.log_artifact(cm_path)
