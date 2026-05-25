import mlflow.pyfunc
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

modelo = {}

class DadosEntrada(BaseModel):
    val_nivelmontante: float = Field(..., example=49.27)
    val_niveljusante:  float = Field(..., example=27.27)
    val_vazaoafluente: float = Field(..., example=1097.0)
    mes:  int = Field(..., example=5)
    dia:  int = Field(..., example=1)
    hora: int = Field(..., example=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns').replace(os.sep, '/')}")
        from mlflow.tracking import MlflowClient
        c = MlflowClient()
        v = c.get_model_version_by_alias("previsor_vazao_balbina", "production")
        source = v.source
        print(f"Carregando modelo de: {source}")
        modelo["pipeline"] = mlflow.pyfunc.load_model(source)
        modelo["uri"] = source
        print("[OK] Modelo carregado!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        modelo["pipeline"] = None
        modelo["uri"] = None
    yield
    modelo.clear()

app = FastAPI(
    title="Previsor de Vazão Defluente - Balbina",
    description="Previsão de vazão defluente da UHE Balbina via GradientBoosting",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/saude")
def saude():
    if modelo.get("pipeline") is None:
        raise HTTPException(status_code=500, detail="Modelo nao carregado.")
    return {"ok": True, "modelo": modelo["uri"]}

@app.post("/predict")
def predict(dados: DadosEntrada):
    if modelo.get("pipeline") is None:
        raise HTTPException(status_code=500, detail="Modelo nao disponivel.")
    import pandas as pd
    df = pd.DataFrame([dados.model_dump()])
    try:
        predicao = modelo["pipeline"].predict(df)
        return {"vazao_defluente_prevista_m3s": round(float(predicao[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))