import os
import shutil
import pickle
from datetime import datetime

DESTINO = "entrega_final"
EXP_ID = "737769590242991027"

IGNORAR = [
    "venv",
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "entrega_final",
    "montar_zip.py",
    "fix_registry.py",
    "find_model.py",
    "check_model.py",
    "cleanup.py",
]

# Limpa destino anterior
if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)
os.makedirs(DESTINO)

# Acha o GradientBoosting mais recente
models_path = os.path.join("mlruns", EXP_ID, "models")
modelos_gb = []
for m in os.listdir(models_path):
    pkl = os.path.join(models_path, m, "artifacts", "model.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            pipe = pickle.load(f)
        if "GradientBoosting" in type(pipe.named_steps['clf']).__name__:
            modelos_gb.append((os.path.getmtime(pkl), m))
modelos_gb.sort(reverse=True)
modelo_mais_recente = modelos_gb[0][1]
print(f"Modelo mais recente: {modelo_mais_recente}")

# Acha o run mais recente de cada tipo
runs_path = os.path.join("mlruns", EXP_ID)
runs_por_tipo = {}
for run_id in os.listdir(runs_path):
    meta = os.path.join(runs_path, run_id, "tags", "mlflow.runName")
    if not os.path.exists(meta):
        continue
    with open(meta) as f:
        nome = f.read().strip()
    meta_yaml = os.path.join(runs_path, run_id, "meta.yaml")
    if os.path.exists(meta_yaml):
        mtime = os.path.getmtime(meta_yaml)
        if nome not in runs_por_tipo or mtime > runs_por_tipo[nome][0]:
            runs_por_tipo[nome] = (mtime, run_id)

runs_manter = {v[1] for v in runs_por_tipo.values()}
print(f"Runs mantidos: {runs_por_tipo.keys()}")

# Copia mlruns limpo
mlruns_dst = os.path.join(DESTINO, "mlruns", EXP_ID)
os.makedirs(mlruns_dst, exist_ok=True)

# Copia meta.yaml do experimento
shutil.copy2(
    os.path.join("mlruns", EXP_ID, "meta.yaml"),
    os.path.join(mlruns_dst, "meta.yaml")
)

# Copia só os runs mais recentes
for run_id in runs_manter:
    src = os.path.join("mlruns", EXP_ID, run_id)
    dst = os.path.join(mlruns_dst, run_id)
    if os.path.exists(src):
        shutil.copytree(src, dst)

# Copia só o modelo mais recente
models_dst = os.path.join(mlruns_dst, "models")
os.makedirs(models_dst, exist_ok=True)
shutil.copytree(
    os.path.join(models_path, modelo_mais_recente),
    os.path.join(models_dst, modelo_mais_recente)
)

# Copia models/previsor_vazao_balbina (registry)
registry_src = os.path.join("mlruns", "models", "previsor_vazao_balbina")
registry_dst = os.path.join(DESTINO, "mlruns", "models", "previsor_vazao_balbina")
if os.path.exists(registry_src):
    shutil.copytree(registry_src, registry_dst)

# Copia o resto do projeto
for item in os.listdir("."):
    if item in IGNORAR or item == "mlruns":
        continue
    src = item
    dst = os.path.join(DESTINO, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "mensais", "__pycache__", ".ipynb_checkpoints"
        ))
    else:
        shutil.copy2(src, dst)

# Calcula tamanho
total = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, dn, fn in os.walk(DESTINO)
    for f in fn
)
print(f"\nPasta '{DESTINO}' criada!")
print(f"Tamanho total: {total / 1024 / 1024:.1f} MB")