# Responsável por carregar o índice de metadados e abrir os GeoTIFFs de NDVI.

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

from config import NDVI_DIR, TALHOES

# Mapeamento inverso: UUID completo → nome legível do talhão
TALHAO_NOMES = {uuid: nome for nome, uuid in TALHOES.items()}


def carregar_metadata(caminho_csv: Path = None) -> pd.DataFrame:

    if caminho_csv is None:
        caminho_csv = NDVI_DIR / "ndvi_metadata.csv"

    df = pd.read_csv(caminho_csv)

    # Extrai o UUID do talhão a partir do nome do arquivo
    # Padrão: ndvi_raw_{UUID}_{YYYY-MM-DD}.tiff
    df["season_id"] = df["filename"].str.extract(r"ndvi_raw_([^_]+(?:-[^_]+){4})_")

    # Mapeia UUID → nome do talhão
    df["talhao_nome"] = df["season_id"].map(TALHAO_NOMES)

    # Extrai a data do nome do arquivo
    df["data"] = pd.to_datetime(
        df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    )

    # Marca imagens completamente nubladas
    df["nublada"] = df["b1_valid_pixels"] == 0

    return df.sort_values(["talhao_nome", "data"]).reset_index(drop=True)


def abrir_tiff(filename: str, pasta: Path = None) -> tuple[np.ndarray, dict]:
    """
    Parâmetros:
      filename : nome do arquivo (ex: 'ndvi_raw_..._2025-08-20.tiff')
      pasta    : diretório onde estão os TIFFs (padrão: NDVI_DIR/ndvi/)

    Retorna:
      ndvi   : array numpy 2D com valores NDVI (float64), shape (40, 54)
      meta   : dicionário com CRS, resolução, dimensões e bounds
    """
    if pasta is None:
        pasta = NDVI_DIR / "ndvi"

    caminho = pasta / filename

    with rasterio.open(caminho) as src:
        ndvi = src.read(1).astype(np.float64)  # Banda 1 = NDVI

        if src.nodata is not None:
            ndvi[ndvi == src.nodata] = np.nan

        meta = {
            "crs":    str(src.crs),
            "res_x":  src.res[0],
            "res_y":  src.res[1],
            "width":  src.width,
            "height": src.height,
            "bounds": src.bounds,
        }

    return ndvi, meta