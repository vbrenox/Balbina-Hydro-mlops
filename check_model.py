import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"file:///{ROOT}/mlruns".replace("\\", "/"))

c = MlflowClient()
v = c.get_model_version_by_alias("previsor_vazao_balbina", "production")
print("Versao:", v.version)
print("Source:", v.source)
print("Status:", v.status)