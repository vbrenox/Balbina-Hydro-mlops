"""
consolidar_balbina.py
---------------------
Lê todos os CSVs mensais hidrológicos da ANEEL de uma pasta,
filtra apenas os registros de BALBINA e salva um único arquivo
consolidado pronto para o pipeline DVC.

Uso:
    python consolidar_balbina.py

    Por padrão espera os CSVs na pasta ./data/raw/mensais/
    e salva o resultado em ./data/raw/balbina_consolidado.csv
"""

import os
import glob
import pandas as pd

# ── Configurações ──────────────────────────────────────────────
PASTA_ENTRADA = "./data/raw/mensais"          # pasta com os 12 CSVs mensais
ARQUIVO_SAIDA = "./data/raw/balbina_consolidado.csv"
FILTRO_USINA  = "BALBINA"                     # valor em nom_reservatorio
SEPARADOR     = ";"                           # ponto e vírgula (padrão ANEEL)
ENCODING      = "latin-1"                     # corrige acentuação da ANEEL
# ──────────────────────────────────────────────────────────────


def carregar_csv(caminho: str) -> pd.DataFrame:
    """Carrega um CSV mensal com tratamento de encoding e separador."""
    try:
        df = pd.read_csv(
            caminho,
            sep=SEPARADOR,
            encoding=ENCODING,
            dtype=str,          # lê tudo como string primeiro (evita surpresas)
            low_memory=False,
        )
        # Remove espaços extras nos nomes das colunas
        df.columns = df.columns.str.strip()
        # Corrige encoding latin-1 → utf-8 nas colunas de texto
        colunas_texto = ["nom_subsistema", "tip_reservatorio", "nom_bacia", "nom_reservatorio"]
        for col in colunas_texto:
            if col in df.columns:
                df[col] = df[col].str.strip().str.encode("latin-1", errors="ignore").str.decode("utf-8", errors="ignore")
        return df
    except Exception as e:
        print(f"  [ERRO] Não foi possível ler {caminho}: {e}")
        return pd.DataFrame()


def filtrar_balbina(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas registros de Balbina (tolerante a espaços extras)."""
    if "nom_reservatorio" not in df.columns:
        print("  [AVISO] Coluna 'nom_reservatorio' não encontrada.")
        return pd.DataFrame()
    return df[df["nom_reservatorio"].str.strip().str.upper() == FILTRO_USINA].copy()


def converter_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas para os tipos corretos após a concatenação."""

    # Data/hora
    if "din_instante" in df.columns:
        df["din_instante"] = pd.to_datetime(
            df["din_instante"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        # Colunas auxiliares úteis para EDA e modelagem
        df["ano"]  = df["din_instante"].dt.year
        df["mes"]  = df["din_instante"].dt.month
        df["dia"]  = df["din_instante"].dt.day
        df["hora"] = df["din_instante"].dt.hour

    # Colunas numéricas
    colunas_numericas = [
        "val_nivelmontante",
        "val_niveljusante",
        "val_volumeutil",
        "val_vazaoafluente",
        "val_vazaodefluente",
        "val_vazaoturbinada",
        "val_vazaovertida",
        "val_vazaooutrasestruturas",
        "val_vazaotransferida",
        "val_vazaovertidanaoturbinavel",
    ]
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "."), errors="coerce"
            )

    return df


def main():
    # Garante que a pasta de saída existe
    os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)

    # Busca todos os CSVs na pasta de entrada
    padrao = os.path.join(PASTA_ENTRADA, "*.csv")
    arquivos = sorted(glob.glob(padrao))

    if not arquivos:
        print(f"[ERRO] Nenhum CSV encontrado em: {PASTA_ENTRADA}")
        return

    print(f"[INFO] {len(arquivos)} arquivo(s) encontrado(s):\n")

    frames = []
    for arq in arquivos:
        nome = os.path.basename(arq)
        print(f"  Processando: {nome}")
        df = carregar_csv(arq)
        if df.empty:
            continue
        df_balbina = filtrar_balbina(df)
        print(f"    → {len(df_balbina)} registros de Balbina encontrados")
        frames.append(df_balbina)

    if not frames:
        print("\n[ERRO] Nenhum registro de Balbina foi encontrado nos arquivos.")
        return

    # Consolida tudo
    df_final = pd.concat(frames, ignore_index=True)
    df_final = converter_tipos(df_final)

    # Ordena por data
    if "din_instante" in df_final.columns:
        df_final = df_final.sort_values("din_instante").reset_index(drop=True)

    # Remove duplicatas (caso haja sobreposição entre arquivos mensais)
    antes = len(df_final)
    df_final = df_final.drop_duplicates(subset=["din_instante"])
    depois = len(df_final)
    if antes != depois:
        print(f"\n[INFO] {antes - depois} duplicatas removidas.")

    # Salva
    df_final.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig", sep=";")

    print(f"\n{'='*50}")
    print(f"[OK] Consolidação concluída!")
    print(f"     Total de registros : {len(df_final):,}")
    print(f"     Período             : {df_final['din_instante'].min()} → {df_final['din_instante'].max()}")
    print(f"     Arquivo salvo em    : {ARQUIVO_SAIDA}")
    print(f"{'='*50}")

    # Preview
    print("\nPrimeiras linhas do dataset consolidado:")
    print(df_final.head(3).to_string())


if __name__ == "__main__":
    main()