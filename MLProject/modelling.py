import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="dataset_preprocessing.csv")
args = parser.parse_args()

# --- Load Dataset ---
df = pd.read_csv(args.dataset_path)
X = df.drop('Personality', axis=1)
y = df['Personality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLflow Experiment ---
mlflow.set_experiment("CI-Workflow-Training")

# --- Model Configurations ---
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 10]
        }
    }
}

# --- Train & Log ---
os.makedirs("saved_models", exist_ok=True)  # root folder to commit later

for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"\n🔍 Tuning {model_name}...")

        # GridSearch
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_acc = cross_val_score(best_model, X_train, y_train, cv=3).mean()

        # Log params & metrics
        mlflow.log_param("model", model_name)
        for param_name, value in grid.best_params_.items():
            mlflow.log_param(param_name, value)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_accuracy", cv_acc)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        model_dir = f"saved_models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        cm_path = f"{model_dir}/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Save & log model + artefak
        model_path = f"{model_dir}/model.pkl"
        joblib.dump(best_model, model_path)

        mlflow.log_artifact(model_path, artifact_path=f"best_model_{model_name}")
        mlflow.log_artifact(cm_path, artifact_path=f"best_model_{model_name}")

        print(f"✅ {model_name} done - Acc: {acc:.4f}, F1: {f1:.4f}, CV Acc: {cv_acc:.4f}")
