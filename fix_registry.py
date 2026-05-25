import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"file:///{ROOT}/mlruns".replace("\\", "/"))

c = MlflowClient()

source = "file:///C:/projetos-estudo/balbina-hydro-mlops/mlruns/737769590242991027/models/m-3a6fd654a83c44b895008f3fb0b58f48/artifacts"

nova = c.create_model_version(
    name="previsor_vazao_balbina",
    source=source,
    run_id="d25a052a8b8b48b4bf1f6b649b6c169e"
)
print(f"Versao criada: {nova.version}")

c.set_registered_model_alias("previsor_vazao_balbina", "production", nova.version)
print(f"Alias @production atribuido a versao {nova.version}!")