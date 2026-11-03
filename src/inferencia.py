# Roda o modelo treinado em uma nova imagem TIFF e retorna máscara + estatísticas

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from config import NDVI_DIR, PROCESSED_DIR, OUTPUTS_DIR, TALHOES, FASES_SAFRA
from modelo import UNet

TARGET_H  = 56
TARGET_W  = 104
DEVICE    = "cpu"
LIMIAR    = 0.5
TALHAO_NOMES = {v: k for k, v in TALHOES.items()}


# ── Utilitários ───────────────────────────────────────────────────────────────

def fase_da_data(data: pd.Timestamp) -> tuple[str, int]:
    """Retorna (nome_fase, índice_fase) para uma data."""
    nomes = list(FASES_SAFRA.keys())
    for i, (nome, (inicio, fim)) in enumerate(FASES_SAFRA.items()):
        if pd.Timestamp(inicio) <= data <= pd.Timestamp(fim):
            return nome, i
    return nomes[0], 0  # fallback: pré-plantio


def carregar_modelo() -> UNet:
    checkpoint = OUTPUTS_DIR / "modelo_melhor.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {checkpoint}\n"
            "Execute src/treino.py antes de rodar a inferência."
        )
    modelo = UNet(in_channels=2).to(DEVICE)
    modelo.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    modelo.eval()
    return modelo


def carregar_baseline(talhao_nome: str, modo: str = "safra") -> tuple:
    """
    Carrega baseline e std do talhão a partir do .npz pré-processado.
    Retorna (baseline, std_global) — arrays (F, H, W) e (H, W).
    """
    npz_path = PROCESSED_DIR / f"ndvi_{modo}_{talhao_nome}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dados pré-processados não encontrados: {npz_path}\n"
            "Execute src/preprocessamento.py primeiro."
        )
    dados = np.load(npz_path)
    cubo     = dados["cubo"].astype(np.float32)
    baseline = dados["baseline"].astype(np.float32)
    std_global = (np.nanstd(cubo, axis=0) + 1e-6).astype(np.float32)
    return baseline, std_global


# ── Inferência principal ───────────────────────────────────────────────────────

def inferir_tiff(caminho_tiff: Path, modelo: UNet = None,
                 modo: str = "safra") -> dict:
    """
    Recebe um caminho de TIFF e retorna dict com máscara e estatísticas.
    """
    # Extrai metadados do nome do arquivo
    stem   = caminho_tiff.stem                        
    partes = stem.split("_")
    data_str    = partes[-1]
    uuid        = "_".join(partes[2:-1])
    talhao_nome = TALHAO_NOMES.get(uuid, uuid[:8])
    data        = pd.Timestamp(data_str)

    # Lê o TIFF
    with rasterio.open(caminho_tiff) as src:
        ndvi   = src.read(1).astype(np.float32)   # Banda 1: NDVI
        mascara_talhao = src.read(3).astype(np.float32)  # Banda 3: área do talhão

    # Verifica se é data nublada
    nublada = np.all(np.isnan(ndvi))
    if nublada:
        return {
            "talhao": talhao_nome, "data": data_str,
            "nublada": True,
            "erro": "Imagem nublada — Banda 1 sem dados NDVI válidos.",
        }

    # Carrega baseline e calcula desvio
    baseline, std_global = carregar_baseline(talhao_nome, modo)
    fase_nome, fase_idx  = fase_da_data(data)
    desvio = (ndvi - baseline[fase_idx]) / std_global

    h_orig, w_orig = ndvi.shape

    # Prepara tensor de entrada
    x = np.stack([ndvi, desvio], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    pad = (0, TARGET_W - x.shape[-1], 0, TARGET_H - x.shape[-2])
    x = F.pad(x, pad, value=0.0)

    # Inferência
    if modelo is None:
        modelo = carregar_modelo()

    with torch.no_grad():
        logits = modelo(x)

    prob  = torch.sigmoid(logits)[0, 0, :h_orig, :w_orig].numpy()
    mascara_pred = (prob > LIMIAR).astype(np.float32)

    # Estatísticas dentro do talhão (Band 3 = 1 → dentro)
    dentro = mascara_talhao > 0
    n_pixels_talhao    = int(dentro.sum())
    n_pixels_anomalos  = int((mascara_pred * dentro).sum())
    pct_anomalia       = 100 * n_pixels_anomalos / max(n_pixels_talhao, 1)

    ndvi_interno       = ndvi[dentro]
    desvio_anomalo     = desvio[mascara_pred.astype(bool) & dentro]

    return {
        "talhao":             talhao_nome,
        "data":               data_str,
        "fase_fenologica":    fase_nome,
        "nublada":            False,
        # Área
        "pixels_talhao":      n_pixels_talhao,
        "pixels_anomalos":    n_pixels_anomalos,
        "pct_anomalia":       round(pct_anomalia, 2),
        # NDVI
        "ndvi_medio":         round(float(np.nanmean(ndvi_interno)), 4),
        "ndvi_min":           round(float(np.nanmin(ndvi_interno)),  4),
        "ndvi_max":           round(float(np.nanmax(ndvi_interno)),  4),
        # Severidade
        "desvio_medio_anomalias": round(float(np.mean(desvio_anomalo)) if len(desvio_anomalo) else 0, 4),
        # Arrays (para visualização e relatório)
        "ndvi_array":         ndvi,
        "desvio_array":       desvio,
        "mascara_array":      mascara_pred,
        "probabilidade":      prob,
    }


def inferir_talhao_data(talhao: str, data: str,
                        modelo: UNet = None, modo: str = "safra") -> dict:
    """Atalho: busca o TIFF pelo talhão e data."""
    uuid = TALHOES.get(talhao)
    if uuid is None:
        raise ValueError(f"Talhão desconhecido: {talhao}")
    caminho = NDVI_DIR / "ndvi" / f"ndvi_raw_{uuid}_{data}.tiff"
    if not caminho.exists():
        raise FileNotFoundError(f"TIFF não encontrado: {caminho}")
    return inferir_tiff(caminho, modelo, modo)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inferência de anomalias NDVI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tiff",   type=str, help="Caminho direto para o arquivo .tiff")
    group.add_argument("--talhao", type=str, help="Nome do talhão (ex: grao_40)")
    parser.add_argument("--data",  type=str, help="Data YYYY-MM-DD (com --talhao)")
    parser.add_argument("--modo",  type=str, default="safra", choices=["safra", "completo"])
    args = parser.parse_args()

    if args.talhao:
        if not args.data:
            parser.error("--data é obrigatório com --talhao")
        resultado = inferir_talhao_data(args.talhao, args.data, modo=args.modo)
    else:
        resultado = inferir_tiff(Path(args.tiff), modo=args.modo)

    # Exibe resumo
    if resultado.get("nublada"):
        print(f"\n[{resultado['data']}] {resultado['talhao']} — IMAGEM NUBLADA")
        return

    print(f"\n{'='*50}")
    print(f"Talhão          : {resultado['talhao']}")
    print(f"Data            : {resultado['data']}")
    print(f"Fase fenológica : {resultado['fase_fenologica']}")
    print(f"NDVI médio      : {resultado['ndvi_medio']}")
    print(f"Anomalia        : {resultado['pct_anomalia']}% dos pixels do talhão")
    print(f"Severidade      : desvio médio nas anomalias = {resultado['desvio_medio_anomalias']:.3f}σ")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()