# Funções de visualização da série temporal NDVI.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import OUTPUTS_DIR
from ndvi_loader import carregar_metadata


def plotar_serie_temporal(salvar: bool = True) -> None:

    df = carregar_metadata()

    NOMES = {
        "grao_40":      "Grão 4.0",
        "grao_conv":    "Grão Convencional",
        "silagem_40":   "Silagem 4.0",
        "silagem_conv": "Silagem Convencional",
    }
    CORES = {
        "grao_40":      "#2d6a4f",
        "grao_conv":    "#52b788",
        "silagem_40":   "#b5651d",
        "silagem_conv": "#e9c46a",
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    for talhao, grupo in df.groupby("talhao_nome"):
        validas  = grupo[~grupo["nublada"]]
        nubladas = grupo[grupo["nublada"]]

        # Linha conectando só os pontos válidos
        ax.plot(
            validas["data"], validas["b1_mean"],
            color=CORES[talhao], linewidth=1.8,
            label=NOMES[talhao]
        )
        # Pontos válidos
        ax.scatter(
            validas["data"], validas["b1_mean"],
            color=CORES[talhao], s=30, zorder=3
        )
        # Marcação das datas nubladas no eixo x
        for data in nubladas["data"]:
            ax.axvline(data, color="#cccccc", linewidth=0.6,
                       linestyle="--", alpha=0.5)

    # Formatação
    ax.set_title("Série Temporal NDVI — FarmLab Safra 2025–2026",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("NDVI médio", fontsize=11)
    ax.set_xlabel("Data", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    if salvar:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUTS_DIR / "ndvi_serie_temporal.png"
        fig.savefig(destino, dpi=150)
        print(f"Gráfico salvo em: {destino}")

    plt.show()


if __name__ == "__main__":
    plotar_serie_temporal()