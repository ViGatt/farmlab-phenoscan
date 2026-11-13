# Gera relatório agronômico em PDF usando LangChain + Gemini Flash

import sys
import argparse
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

load_dotenv()

from config import TALHOES, FASES_SAFRA
from inferencia import inferir_talhao_data   # ← chama direto, sem .npz

# ── Configurações ─────────────────────────────────────────────────────────────

MODELO_LLM = "gemini-3.8-flash"
OUTPUT_DIR = Path("outputs/relatorios")

NOMES_TALHOES = {
    "silagem_conv": "Silagem Convencional",
    "grao_40":      "Grão 40 (Regenerativo)",
    "grao_conv":    "Grão Convencional",
    "silagem_40":   "Silagem 40 (Regenerativo)",
}

# ── Geração de texto com LLM ──────────────────────────────────────────────────

PROMPT_TEMPLATE = """Você é um agrônomo especialista em agricultura de precisão e sensoriamento remoto.
Analise os dados NDVI de satélite do talhão abaixo e gere um relatório técnico em português.

INFORMAÇÕES DO TALHÃO:
- Nome: {nome_talhao}
- Data de análise: {data}
- Fase fenológica: {fase}

MÉTRICAS NDVI:
- NDVI médio: {ndvi_medio:.4f}
- NDVI mínimo: {ndvi_min:.4f}
- NDVI máximo: {ndvi_max:.4f}
- Percentual de anomalias detectadas: {pct_anomalia:.1f}%
- Desvio médio nas anomalias: {desvio_anomalia:.3f}σ
- Pixels no talhão: {pixels_talhao}
- Pixels anômalos: {pixels_anomalos}

Gere um relatório técnico estruturado com exatamente estas seções:

[RESUMO]
Resumo executivo em 2-3 frases sobre a condição geral do talhão.

[ANALISE]
Análise técnica detalhada dos dados NDVI, interpretando o que os valores indicam sobre o desenvolvimento da cultura de milho nesta fase fenológica. Comente sobre a distribuição espacial das anomalias e possíveis causas agronômicas.

[RECOMENDACOES]
Lista de 3 a 5 recomendações práticas para o produtor com base nos dados observados.

Seja técnico, objetivo e use terminologia agronômica adequada."""


def gerar_texto_llm(resultado: dict) -> dict:
    """Gera texto analítico com Gemini via LangChain."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não encontrada no .env")

    llm = ChatGoogleGenerativeAI(
        model=MODELO_LLM,
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({
        "nome_talhao":     NOMES_TALHOES.get(resultado["talhao"], resultado["talhao"]),
        "data":            resultado["data"],
        "fase":            resultado["fase_fenologica"],
        "ndvi_medio":      resultado["ndvi_medio"],
        "ndvi_min":        resultado["ndvi_min"],
        "ndvi_max":        resultado["ndvi_max"],
        "pct_anomalia":    resultado["pct_anomalia"],
        "desvio_anomalia": resultado["desvio_medio_anomalias"],
        "pixels_talhao":   resultado["pixels_talhao"],
        "pixels_anomalos": resultado["pixels_anomalos"],
    })

    texto = response.content
    # langchain_google_genai pode retornar lista de blocos
    if isinstance(texto, list):
        texto = "".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in texto
        )

    # Separar seções
    secoes = {}
    for secao in ["RESUMO", "ANALISE", "RECOMENDACOES"]:
        inicio_tag = f"[{secao}]"
        idx_inicio = texto.find(inicio_tag)
        if idx_inicio == -1:
            secoes[secao] = ""
            continue
        idx_inicio += len(inicio_tag)
        idx_fim = len(texto)
        for outra in ["RESUMO", "ANALISE", "RECOMENDACOES"]:
            if outra == secao:
                continue
            pos = texto.find(f"[{outra}]", idx_inicio)
            if pos != -1 and pos < idx_fim:
                idx_fim = pos
        secoes[secao] = texto[idx_inicio:idx_fim].strip()

    return secoes


# ── Geração do PDF ────────────────────────────────────────────────────────────

def gerar_pdf(resultado: dict, secoes: dict) -> Path:
    """Gera o relatório em PDF com ReportLab."""
    talhao = resultado["talhao"]
    data   = resultado["data"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    caminho_pdf = OUTPUT_DIR / f"relatorio_{talhao}_{data}.pdf"

    doc = SimpleDocTemplate(
        str(caminho_pdf),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
    )

    verde_escuro = HexColor("#1B5E20")
    verde_medio  = HexColor("#388E3C")
    cinza_fundo  = HexColor("#F5F5F5")

    styles = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        "Titulo", parent=styles["Title"],
        fontSize=18, textColor=verde_escuro, spaceAfter=6,
    )
    estilo_subtitulo = ParagraphStyle(
        "Subtitulo", parent=styles["Normal"],
        fontSize=11, textColor=verde_medio, spaceAfter=12,
    )
    estilo_secao = ParagraphStyle(
        "Secao", parent=styles["Heading2"],
        fontSize=13, textColor=verde_escuro,
        spaceBefore=14, spaceAfter=6,
    )
    estilo_corpo = ParagraphStyle(
        "Corpo", parent=styles["Normal"],
        fontSize=10, leading=15,
        alignment=TA_JUSTIFY, spaceAfter=8,
    )
    estilo_rodape = ParagraphStyle(
        "Rodape", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#757575"),
        alignment=TA_CENTER,
    )

    elementos = []

    # Cabeçalho
    elementos.append(Paragraph("FarmLab PhenoScan", estilo_titulo))
    elementos.append(Paragraph(
        f"Relatório NDVI — {NOMES_TALHOES.get(talhao, talhao)} | {data}",
        estilo_subtitulo,
    ))
    elementos.append(Spacer(1, 0.3*cm))

    # Tabela de métricas
    dados_tabela = [
        ["Métrica", "Valor"],
        ["NDVI Médio",             f"{resultado['ndvi_medio']:.4f}"],
        ["NDVI Mínimo",            f"{resultado['ndvi_min']:.4f}"],
        ["NDVI Máximo",            f"{resultado['ndvi_max']:.4f}"],
        ["Anomalias Detectadas",   f"{resultado['pct_anomalia']:.1f}%"],
        ["Desvio nas Anomalias",   f"{resultado['desvio_medio_anomalias']:.3f}σ"],
        ["Fase Fenológica",        resultado["fase_fenologica"].replace("_", " ").title()],
        ["Pixels no Talhão",       str(resultado["pixels_talhao"])],
    ]

    tabela = Table(dados_tabela, colWidths=[9*cm, 7*cm])
    tabela.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  verde_escuro),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  HexColor("#FFFFFF")),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [HexColor("#FFFFFF"), cinza_fundo]),
        ("GRID",          (0, 0), (-1, -1), 0.5, HexColor("#BDBDBD")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elementos.append(tabela)
    elementos.append(Spacer(1, 0.4*cm))

    # Seções de texto
    mapa_secoes = {
        "RESUMO":        "Resumo Executivo",
        "ANALISE":       "Análise Técnica",
        "RECOMENDACOES": "Recomendações",
    }
    for chave, titulo_secao in mapa_secoes.items():
        texto = secoes.get(chave, "").strip()
        if not texto:
            continue
        elementos.append(Paragraph(titulo_secao, estilo_secao))
        for paragrafo in texto.split("\n"):
            paragrafo = paragrafo.strip()
            if paragrafo:
                elementos.append(Paragraph(paragrafo, estilo_corpo))

    elementos.append(Spacer(1, 1*cm))
    elementos.append(Paragraph(
        f"Gerado pelo FarmLab PhenoScan em {datetime.now().strftime('%d/%m/%Y %H:%M')} | "
        "UNIMAR × Jacto — Safra 2025/2026",
        estilo_rodape,
    ))

    doc.build(elementos)
    return caminho_pdf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Relatório PDF com análise NDVI + LLM")
    parser.add_argument("--talhao", required=True, choices=list(TALHOES.keys()))
    parser.add_argument("--data",   required=True, help="Data YYYY-MM-DD")
    parser.add_argument("--modo",   default="safra", choices=["safra", "completo"])
    args = parser.parse_args()

    print(f"[relatorio] Talhão : {args.talhao}")
    print(f"[relatorio] Data   : {args.data}")

    print("[relatorio] Rodando inferência...")
    resultado = inferir_talhao_data(args.talhao, args.data, modo=args.modo)

    if resultado.get("nublada"):
        print(f"[relatorio] ERRO: imagem nublada em {args.data}, sem dados NDVI.")
        return

    print(f"[relatorio] NDVI médio: {resultado['ndvi_medio']} | Anomalia: {resultado['pct_anomalia']}%")

    print("[relatorio] Gerando análise com Gemini Flash...")
    secoes = gerar_texto_llm(resultado)

    print("[relatorio] Montando PDF...")
    caminho = gerar_pdf(resultado, secoes)

    print(f"[relatorio] ✓ PDF gerado: {caminho}")


if __name__ == "__main__":
    main()