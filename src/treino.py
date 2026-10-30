# Loop de treinamento LOFO: treina em 3 talhões, testa em grao_40

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config import PROCESSED_DIR, OUTPUTS_DIR, TALHOES
from modelo import UNet, perda_combinada

# ── Configurações ─────────────────────────────────────────────────────────────
TALHAO_TESTE = "grao_40"
MODO         = "safra"        # "safra" ou "completo"
EPOCHS       = 50
BATCH_SIZE   = 8
LR           = 1e-3
VAL_FRAC     = 0.15           # fração dos dados de treino usada como validação
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
TARGET_H, TARGET_W = 56, 104   # tamanho comum (divisível por 8)

torch.manual_seed(SEED)
np.random.seed(SEED)

TALHAO_NOMES = list(TALHOES.keys())
TALHOES_TREINO = [t for t in TALHAO_NOMES if t != TALHAO_TESTE]


# ── Dataset ───────────────────────────────────────────────────────────────────

class NDVIDataset(Dataset):
    """
    Cada amostra = uma data de um talhão.
    x : (2, 40, 54)  — canal 0: NDVI | canal 1: desvio do baseline
    y : (1, 40, 54)  — máscara binária de anomalia
    """
    def __init__(self, talhoes: list, modo: str, apenas_safra: bool = True):
        self.amostras = []

        for nome in talhoes:
            npz_ndvi   = PROCESSED_DIR / f"ndvi_{modo}_{nome}.npz"
            npz_rotulo = PROCESSED_DIR / f"rotulos_{modo}_{nome}.npz"

            if not npz_ndvi.exists() or not npz_rotulo.exists():
                print(f"  [aviso] arquivos não encontrados para {nome}, pulando.")
                continue

            d_ndvi   = np.load(npz_ndvi)
            d_rotulo = np.load(npz_rotulo)

            cubo     = d_ndvi["cubo"].astype(np.float32)    # (N, 40, 54)
            desvio   = d_ndvi["desvio"].astype(np.float32)  # (N, 40, 54)
            mascaras = d_rotulo["mascaras"].astype(np.float32)  # (N, 40, 54)
            em_safra = d_ndvi["em_safra"]                    # (N,) bool

            indices = np.where(em_safra)[0] if apenas_safra else np.arange(len(em_safra))

            for i in indices:
                x = np.stack([cubo[i], desvio[i]], axis=0)  # (2, 40, 54)
                y = mascaras[i][np.newaxis]                  # (1, 40, 54)
                self.amostras.append((x, y))

        print(f"  Dataset '{talhoes}': {len(self.amostras)} amostras")

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        x, y = self.amostras[idx]
        x = torch.tensor(x)   # (2, H, W)
        y = torch.tensor(y)   # (1, H, W)

        # Padding para tamanho comum (zeros nas bordas)
        _, h, w = x.shape
        pad = (0, TARGET_W - w, 0, TARGET_H - h)  # (esq, dir, cima, baixo)
        x = torch.nn.functional.pad(x, pad, value=0.0)
        y = torch.nn.functional.pad(y, pad, value=0.0)
        return x, y


# ── Métricas ──────────────────────────────────────────────────────────────────

def dice_score(pred_logits, alvo, limiar=0.5):
    pred = (torch.sigmoid(pred_logits) > limiar).float()
    alvo = alvo.float()
    inter = (pred * alvo).sum()
    union = pred.sum() + alvo.sum()
    if union == 0:
        return torch.tensor(1.0)
    return (2 * inter / union).item()


def iou_score(pred_logits, alvo, limiar=0.5):
    pred = (torch.sigmoid(pred_logits) > limiar).float()
    alvo = alvo.float()
    inter = (pred * alvo).sum()
    union = pred.sum() + alvo.sum() - inter
    if union == 0:
        return torch.tensor(1.0)
    return (inter / union).item()


# ── Treino ────────────────────────────────────────────────────────────────────

def treinar():
    print(f"\nDispositivo: {DEVICE}")
    print(f"Talhões de treino : {TALHOES_TREINO}")
    print(f"Talhão de teste   : {TALHAO_TESTE}\n")

    # Datasets
    ds_total = NDVIDataset(TALHOES_TREINO, MODO, apenas_safra=True)
    n_val    = max(1, int(len(ds_total) * VAL_FRAC))
    n_treino = len(ds_total) - n_val
    ds_treino, ds_val = random_split(ds_total, [n_treino, n_val],
                                     generator=torch.Generator().manual_seed(SEED))

    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True)
    dl_val    = DataLoader(ds_val,    batch_size=BATCH_SIZE, shuffle=False)

    ds_teste = NDVIDataset([TALHAO_TESTE], MODO, apenas_safra=True)
    dl_teste = DataLoader(ds_teste, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    modelo    = UNet(in_channels=2).to(DEVICE)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=5)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    melhor_dice = 0.0
    melhor_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # — Treino —
        modelo.train()
        perda_ep = 0.0
        for x, y in dl_treino:
            x, y = x.to(DEVICE), y.to(DEVICE)
            otimizador.zero_grad()
            pred = modelo(x)
            perda = perda_combinada(pred, y)
            perda.backward()
            otimizador.step()
            perda_ep += perda.item()

        perda_ep /= len(dl_treino)

        # — Validação —
        modelo.eval()
        dice_val = 0.0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = modelo(x)
                dice_val += dice_score(pred, y)
        dice_val /= len(dl_val)

        scheduler.step(dice_val)

        if dice_val > melhor_dice:
            melhor_dice  = dice_val
            melhor_epoch = epoch
            torch.save(modelo.state_dict(), OUTPUTS_DIR / "modelo_melhor.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"perda: {perda_ep:.4f} | "
                  f"dice_val: {dice_val:.4f} | "
                  f"melhor: {melhor_dice:.4f} (epoch {melhor_epoch})")

    # — Teste final no talhão reservado —
    print(f"\n=== Teste final: {TALHAO_TESTE} ===")
    modelo.load_state_dict(torch.load(OUTPUTS_DIR / "modelo_melhor.pt"))
    modelo.eval()

    dice_t, iou_t, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in dl_teste:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = modelo(x)
            dice_t += dice_score(pred, y)
            iou_t  += iou_score(pred, y)
            n += 1

    print(f"Dice Score : {dice_t/n:.4f}")
    print(f"IoU Score  : {iou_t/n:.4f}")
    print(f"\nModelo salvo em: {OUTPUTS_DIR / 'modelo_melhor.pt'}")


if __name__ == "__main__":
    treinar()