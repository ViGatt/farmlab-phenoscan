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
    "silagem_conv": "0bf86c8b",
    "grao_40":      "258e47e0",
    "grao_conv":    "b7292cb8",
    "silagem_40":   "f791bf13",
}

# ── Safra ────────────────────────────────────────────────────────────────────
SAFRA_INICIO = "2025-08-01"
SAFRA_FIM    = "2026-05-01"