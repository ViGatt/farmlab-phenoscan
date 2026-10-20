# Configurações centrais do projeto farmlab-phenoscan

from pathlib import Path

# ── Raiz do projeto ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Dados brutos  ────────────────────────────────────────
RAW_DIR        = ROOT / "data" / "raw"
NDVI_DIR       = RAW_DIR / "onesoil"       # GeoTIFFs + CSVs do OneSoil
METOS_DIR      = RAW_DIR / "metos"         # CSV/Parquet da estação meteorológica
CROPMAN_DIR    = RAW_DIR / "cropman"       # Análise de solo
EKOS_DIR       = RAW_DIR / "ekos"          # Telemetria e armadilhas

# ── Dados processados ────────────────────────────────────────────────────────
PROCESSED_DIR  = ROOT / "data" / "processed"

# ── Saídas do modelo ─────────────────────────────────────────────────────────
OUTPUTS_DIR    = ROOT / "outputs"

# ── Talhões do FarmLab (IDs do OneSoil) ─────────────────────────────────────
TALHOES = {
    "silagem_conv": "0bf86c8b-2779-4a98-ba25-7cf9518ee316",
    "grao_40":      "258e47e0-3bdc-4419-bedd-7b85812c2113",
    "grao_conv":    "b7292cb8-72bb-447a-8feb-8ac983afd50b",
    "silagem_40":   "f791bf13-1d24-4b4f-88bd-2569162df2b3",
}

# ── Safra ────────────────────────────────────────────────────────────────────
SAFRA_INICIO = "2025-08-01"
SAFRA_FIM    = "2026-05-01"