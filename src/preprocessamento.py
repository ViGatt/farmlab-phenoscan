# Pré-processamento NDVI: gap-filling + baseline fenológico

# Modo SAFRA    → 107 datas, filtra até SAFRA_FIM (~mai/2026)
# Modo COMPLETO → 107 datas, usa todo o período disponível

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import rasterio

from config import (
    NDVI_DIR, PROCESSED_DIR, TALHOES,
    SAFRA_INICIO, SAFRA_FIM, PERIODO_COMPLETO_FIM,
    FASES_SAFRA, FASES_EXTENDIDO,
)

TALHAO_NOMES = {v: k for k, v in TALHOES.items()}


def fase_da_data(data: pd.Timestamp, fases: dict) -> str:
    for nome, (inicio, fim) in fases.items():
        if pd.Timestamp(inicio) <= data <= pd.Timestamp(fim):
            return nome
    return list(fases.keys())[0]


def carregar_tiffs(pasta_ndvi: Path, data_min: str, data_max: str) -> dict:
    """Carrega todos os TIFFs dentro do intervalo de datas."""
    arquivos = sorted(pasta_ndvi.glob("ndvi_raw_*.tiff"))
    print(f"TIFFs encontrados na pasta: {len(arquivos)}")

    t_min = pd.Timestamp(data_min)
    t_max = pd.Timestamp(data_max)

    grupos: dict[str, list] = {}

    for arq in arquivos:
        partes = arq.stem.split("_")
        data_str = partes[-1]
        uuid = "_".join(partes[2:-1])

        try:
            data = pd.Timestamp(data_str)
        except Exception:
            continue

        if not (t_min <= data <= t_max):
            continue

        with rasterio.open(arq) as src:
            b1 = src.read(1).astype(np.float64)

        if uuid not in grupos:
            grupos[uuid] = []
        grupos[uuid].append((data, b1))

    for uuid in grupos:
        grupos[uuid].sort(key=lambda x: x[0])

    return grupos


def gap_fill(cubo_bruto: np.ndarray, idx_validos: np.ndarray) -> np.ndarray:
    """Interpolação linear pixel a pixel para datas nubladas."""
    n = cubo_bruto.shape[0]
    todos = np.arange(n)
    resultado = np.empty_like(cubo_bruto)
    for row in range(cubo_bruto.shape[1]):
        for col in range(cubo_bruto.shape[2]):
            vals = cubo_bruto[idx_validos, row, col]
            resultado[:, row, col] = np.interp(todos, idx_validos, vals)
    return resultado


def calcular_baseline(cubo: np.ndarray, datas: list, fases: dict) -> np.ndarray:
    """Mediana por pixel por fase fenológica."""
    nomes = list(fases.keys())
    baseline = np.zeros((len(nomes), cubo.shape[1], cubo.shape[2]))
    for i, nome in enumerate(nomes):
        idx = [j for j, d in enumerate(datas) if fase_da_data(d, fases) == nome]
        if idx:
            baseline[i] = np.nanmedian(cubo[idx], axis=0)
    return baseline


def calcular_desvio(cubo: np.ndarray, datas: list, baseline: np.ndarray, fases: dict) -> np.ndarray:
    """Desvio normalizado em relação ao baseline da fase fenológica."""
    std_global = np.nanstd(cubo, axis=0) + 1e-6
    nomes = list(fases.keys())
    desvio = np.zeros_like(cubo)
    for j, d in enumerate(datas):
        fase_idx = nomes.index(fase_da_data(d, fases))
        desvio[j] = (cubo[j] - baseline[fase_idx]) / std_global
    return desvio


def processar_talhao(uuid: str, registros: list, talhao_nome: str,
                     fases: dict, modo: str, safra_fim: str) -> None:
    datas  = [r[0] for r in registros]
    arrays = [r[1] for r in registros]
    n      = len(datas)

    cubo_bruto = np.stack(arrays, axis=0)
    nubladas   = np.array([np.all(np.isnan(a)) for a in arrays])
    idx_validos = np.where(~nubladas)[0]

    print(f"  [{talhao_nome}] {n} datas | "
          f"{(~nubladas).sum()} válidas | {nubladas.sum()} nubladas")

    if len(idx_validos) < 2:
        print(f"  [{talhao_nome}] AVISO: dados insuficientes, pulando.")
        return

    cubo_filled = gap_fill(cubo_bruto, idx_validos)
    baseline    = calcular_baseline(cubo_filled, datas, fases)
    desvio      = calcular_desvio(cubo_filled, datas, baseline, fases)

    t_safra_fim = pd.Timestamp(safra_fim)
    em_safra    = np.array([d <= t_safra_fim for d in datas])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destino = PROCESSED_DIR / f"ndvi_{modo}_{talhao_nome}.npz"
    np.savez_compressed(
        destino,
        cubo     = cubo_filled,
        baseline = baseline,
        desvio   = desvio,
        datas    = np.array([str(d.date()) for d in datas]),
        nubladas = nubladas,
        em_safra = em_safra,
    )
    print(f"  [{talhao_nome}] Salvo → {destino} | shape: {cubo_filled.shape}")


def processar(modo: str = "safra") -> None:
    """
    modo='safra'    → só a safra 2025/2026 (até SAFRA_FIM)
    modo='completo' → todo o período disponível
    """
    assert modo in ("safra", "completo"), "modo deve ser 'safra' ou 'completo'"

    pasta_ndvi = NDVI_DIR / "ndvi"
    if not pasta_ndvi.exists():
        print(f"ERRO: pasta não encontrada: {pasta_ndvi}")
        return

    if modo == "safra":
        data_min, data_max = SAFRA_INICIO, SAFRA_FIM
        fases = FASES_SAFRA
    else:
        data_min, data_max = SAFRA_INICIO, PERIODO_COMPLETO_FIM
        fases = FASES_EXTENDIDO

    print(f"\n=== Modo: {modo.upper()} | {data_min} → {data_max} ===")

    grupos = carregar_tiffs(pasta_ndvi, data_min, data_max)

    for uuid, registros in grupos.items():
        nome = TALHAO_NOMES.get(uuid, uuid[:8])
        processar_talhao(uuid, registros, nome, fases, modo, SAFRA_FIM)

    print(f"\nPré-processamento '{modo}' concluído.")


if __name__ == "__main__":
    processar("safra")      # Cenário 1: safra principal
    processar("completo")   # Cenário 2: período estendido