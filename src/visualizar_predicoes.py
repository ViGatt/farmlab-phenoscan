# Valida visualmente as predições do modelo no talhão de teste (grao_40)
# Painéis: JPG OneSoil | NDVI atual | Desvio | Máscara real | Máscara predita

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import torch.nn.functional as F

from config import PROCESSED_DIR, OUTPUTS_DIR, NDVI_DIR, TALHOES
from modelo import UNet

# ── Configurações ─────────────────────────────────────────────────────────────
TALHAO     = "grao_40"
MODO       = "safra"
N_ANOMALAS = 6
N_NORMAIS  = 3
TARGET_H   = 56
TARGET_W   = 104
DEVICE     = "cpu"
LIMIAR     = 0.5

JPG_DIR    = NDVI_DIR / "jpg"
UUID_TALHAO = TALHOES[TALHAO]


def carregar_modelo() -> UNet:
    checkpoint = OUTPUTS_DIR / "modelo_melhor.pt"
    modelo = UNet(in_channels=2).to(DEVICE)
    modelo.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    modelo.eval()
    return modelo


def inferir(modelo, ndvi, desvio, h_orig, w_orig):
    x = np.stack([ndvi, desvio], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    pad = (0, TARGET_W - x.shape[-1], 0, TARGET_H - x.shape[-2])
    x = F.pad(x, pad, value=0.0)
    with torch.no_grad():
        logits = modelo(x)
    pred = (torch.sigmoid(logits) > LIMIAR).float()
    return pred[0, 0, :h_orig, :w_orig].numpy()


def plotar(data, ndvi, desvio, mascara_real, mascara_pred, destino):
    jpg_path = JPG_DIR / f"ndvi_raw_{UUID_TALHAO}_{data}.jpg"
    tem_jpg  = jpg_path.exists()

    n_paineis = 5 if tem_jpg else 4
    fig, axes = plt.subplots(1, n_paineis, figsize=(4 * n_paineis, 4))
    fig.suptitle(f"{TALHAO} — {data}", fontsize=13, fontweight="bold", y=1.02)

    col = 0

    # Painel 0 (opcional): JPG OneSoil
    if tem_jpg:
        img = mpimg.imread(str(jpg_path))
        axes[col].imshow(img)
        axes[col].set_title("Imagem OneSoil (JPG)")
        col += 1

    # Painel: NDVI atual
    im1 = axes[col].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[col].set_title("NDVI atual")
    plt.colorbar(im1, ax=axes[col], fraction=0.046)
    col += 1

    # Painel: Desvio do baseline
    im2 = axes[col].imshow(desvio, cmap="RdBu", vmin=-3, vmax=3)
    axes[col].set_title("Desvio do baseline")
    plt.colorbar(im2, ax=axes[col], fraction=0.046)
    col += 1

    # Patches para legenda
    patch_a = mpatches.Patch(color="red",   label="Anômalo")
    patch_n = mpatches.Patch(facecolor="white", edgecolor="gray", label="Normal")

    # Painel: Máscara real
    axes[col].imshow(mascara_real, cmap="Reds", vmin=0, vmax=1)
    axes[col].set_title("Anomalia real (rótulo)")
    axes[col].set_xlabel(f"{100*mascara_real.mean():.1f}% anômalo", fontsize=8)
    axes[col].legend(handles=[patch_a, patch_n], loc="lower right", fontsize=7)
    col += 1

    # Painel: Máscara predita
    axes[col].imshow(mascara_pred, cmap="Reds", vmin=0, vmax=1)
    axes[col].set_title("Anomalia predita (modelo)")
    axes[col].set_xlabel(f"{100*mascara_pred.mean():.1f}% anômalo", fontsize=8)
    axes[col].legend(handles=[patch_a, patch_n], loc="lower right", fontsize=7)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {'✓ JPG' if tem_jpg else '  sem JPG'} | {destino.name}")


def visualizar():
    d_ndvi   = np.load(PROCESSED_DIR / f"ndvi_{MODO}_{TALHAO}.npz")
    d_rotulo = np.load(PROCESSED_DIR / f"rotulos_{MODO}_{TALHAO}.npz")

    cubo     = d_ndvi["cubo"].astype(np.float32)
    desvio   = d_ndvi["desvio"].astype(np.float32)
    mascaras = d_rotulo["mascaras"].astype(np.float32)
    datas    = d_ndvi["datas"]
    em_safra = d_ndvi["em_safra"]

    idx_safra    = np.where(em_safra)[0]
    pct_anomalas = np.array([mascaras[i].mean() for i in idx_safra])
    ord_desc     = np.argsort(pct_anomalas)[::-1]

    idx_selec = (
        [idx_safra[i] for i in ord_desc[:N_ANOMALAS]] +
        [idx_safra[i] for i in ord_desc[-N_NORMAIS:]]
    )

    h_orig, w_orig = cubo.shape[1], cubo.shape[2]
    modelo = carregar_modelo()

    print(f"\n=== Visualizando {TALHAO} ({len(idx_selec)} datas) ===")

    for i in idx_selec:
        data   = str(datas[i])
        pred   = inferir(modelo, cubo[i], desvio[i], h_orig, w_orig)
        destino = OUTPUTS_DIR / "predicoes" / f"{TALHAO}_{data}.png"
        plotar(data, cubo[i], desvio[i], mascaras[i], pred, destino)

    print(f"\nFiguras salvas em: outputs/predicoes/")


if __name__ == "__main__":
    visualizar()