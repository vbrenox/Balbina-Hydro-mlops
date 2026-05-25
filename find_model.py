import os
import pickle
from datetime import datetime

base = "mlruns/737769590242991027/models"
for m in os.listdir(base):
    pkl_path = os.path.join(base, m, "artifacts", "model.pkl")
    mod_time = os.path.getmtime(pkl_path)
    with open(pkl_path, "rb") as f:
        modelo = pickle.load(f)
    nome = type(modelo.named_steps['clf']).__name__
    print(f"{datetime.fromtimestamp(mod_time)} | {m} | {nome}")