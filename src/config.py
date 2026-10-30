from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT / "data" / "raw"
NDVI_DIR      = RAW_DIR / "onesoil"
METOS_DIR     = RAW_DIR / "metos"
CROPMAN_DIR   = RAW_DIR / "cropman"
EKOS_DIR      = RAW_DIR / "ekos"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR   = ROOT / "outputs"

TALHOES = {
    "silagem_conv": "0bf86c8b-2779-4a98-ba25-7cf9518ee316",
    "grao_40":      "258e47e0-3bdc-4419-bedd-7b85812c2113",
    "grao_conv":    "b7292cb8-72bb-447a-8feb-8ac983afd50b",
    "silagem_40":   "f791bf13-1d24-4b4f-88bd-2569162df2b3",
}

# Safra principal 2025/2026
SAFRA_INICIO = "2025-08-01"
SAFRA_FIM    = "2026-05-15"   # ~colheita

# Período completo disponível (pós-colheita e início de eventual nova safra)
PERIODO_COMPLETO_FIM = "2026-10-31"

# Fases fenológicas da safra 2025/2026
# Ajustadas com base na série temporal real observada
FASES_SAFRA = {
    "pre_plantio":   ("2025-08-01", "2025-10-14"),
    "crescimento":   ("2025-10-15", "2026-01-14"),
    "florescimento": ("2026-01-15", "2026-03-31"),
    "colheita":      ("2026-04-01", "2026-05-15"),
}

# Fases do período estendido (pós-colheita)
FASES_EXTENDIDO = {
    "pre_plantio":   ("2025-08-01", "2025-10-14"),
    "crescimento":   ("2025-10-15", "2026-01-14"),
    "florescimento": ("2026-01-15", "2026-03-31"),
    "colheita":      ("2026-04-01", "2026-05-15"),
    "pos_colheita":  ("2026-05-16", "2026-10-31"),
}