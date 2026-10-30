# Gera máscaras binárias de anomalia a partir dos cubos pré-processados. (
# pixel com desvio < -1.5σ do baseline fenológico = anomalia)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from config import PROCESSED_DIR, TALHOES

TALHAO_NOMES = {v: k for k, v in TALHOES.items()}
LIMIAR = -1.5 


def rotular(modo: str = "safra") -> None:
    assert modo in ("safra", "completo")
    print(f"\n=== Rotulagem | modo: {modo.upper()} ===")

    for nome in TALHOES.keys():
        caminho = PROCESSED_DIR / f"ndvi_{modo}_{nome}.npz"
        if not caminho.exists():
            print(f"  [{nome}] arquivo não encontrado, pulando.")
            continue

        dados = np.load(caminho)
        desvio   = dados["desvio"]    # (N, 40, 54)
        datas    = dados["datas"]
        em_safra = dados["em_safra"]

        mascaras = desvio < LIMIAR    

        total_pixels   = mascaras.size
        pixels_anomalos = mascaras.sum()
        pct = 100 * pixels_anomalos / total_pixels

        print(f"  [{nome}] {len(datas)} datas | "
              f"{pixels_anomalos:,} pixels anômalos ({pct:.1f}%)")

        destino = PROCESSED_DIR / f"rotulos_{modo}_{nome}.npz"
        np.savez_compressed(
            destino,
            mascaras = mascaras,
            datas    = datas,
            em_safra = em_safra,
        )
        print(f"  [{nome}] Salvo → {destino}")

    print(f"\nRotulagem '{modo}' concluída.")


if __name__ == "__main__":
    rotular("safra")
    rotular("completo")