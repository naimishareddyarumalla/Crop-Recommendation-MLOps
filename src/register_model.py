import mlflow
import mlflow.sklearn
import joblib
import os

MODEL_PATH = "models/crop_recommendation_model.pkl"
MODEL_NAME = "Crop_Recommendation_Model"

def register_model():
    print("[INFO] Starting Model Registry")

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Trained model not found!")
        return

    model = joblib.load(MODEL_PATH)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Crop_Recommendation_MLOps")

    with mlflow.start_run(run_name="Model_Registry"):

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format="pickle"
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        print(f"[SUCCESS] Model Registered: {MODEL_NAME}")
        print(f"[INFO] Model Version: {registered_model.version}")

    print("[SUCCESS] Model Registry Completed!")

if __name__ == "__main__":
    register_model()