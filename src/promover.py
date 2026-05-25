import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///C:/projetos-estudo/balbina-hydro-mlops/mlruns")

c = MlflowClient()
c.set_registered_model_alias("previsor_vazao_balbina", "production", "6")
print("Alias @production atribuido a versao 6!")