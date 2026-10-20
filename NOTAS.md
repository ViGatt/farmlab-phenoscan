# Notas de Desenvolvimento — farmlab-phenoscan

## Configuração inicial (Out/2026)

Comecei o repositório do zero. Escolhi usar conda ao invés de venv porque
vou trabalhar com GeoTIFFs (formato de imagem de satélite), e a biblioteca
necessária pra isso (rasterio) depende do GDAL — que no Windows não instala
direito via pip. O conda resolve isso automaticamente.

Ambiente: Python 3.11, rodando no Windows com Anaconda.

---

## Estrutura dos dados (OneSoil / Sentinel-2)

Os dados de satélite vêm da plataforma OneSoil, que processa imagens do
Sentinel-2 (~10m por pixel). São 196 imagens no total: 4 talhões × 49 datas,
cobrindo agosto de 2025 a fevereiro de 2026.

Cada arquivo segue o padrão:
`ndvi_raw_{UUID-do-talhão}_{YYYY-MM-DD}.tiff`

Cada imagem tem 3 bandas:
- Banda 1: valor NDVI do pixel (-1 a 1)
- Banda 2: máscara de qualidade (0 = pixel válido)
- Banda 3: máscara do talhão (delimita a área agrícola)

O arquivo `ndvi_metadata.csv` funciona como índice — tem 196 linhas com
estatísticas de cada imagem: NDVI médio, mín, máx, desvio padrão, e
porcentagem de cada classe de vegetação.

---

## Primeira análise da série temporal

Ao carregar e plotar o NDVI médio ao longo da safra, o resultado fez sentido
agronômico logo de cara:

- **Ago–Out/2025**: NDVI caindo de ~0.6 para ~0.2 — solo em preparação
  antes do plantio
- **Out–Dez/2025**: valores baixos e estáveis — plantio e germinação
- **Jan/2026**: subida rápida para 0.6–0.8 — crescimento vegetativo intenso
- **Fev/2026**: pico da safra (~0.75–0.85) — florescimento

Uma queda brusca aparece em janeiro, simultânea nos 4 talhões. Pode ser
um evento real de estresse ou passagem de nuvem parcial — vai ser
interessante investigar isso depois com os dados meteorológicos do Metos.

Das 196 imagens, 119 são válidas e 77 estão completamente nubladas.
Essas datas nubladas precisam ser preenchidas por interpolação antes
de qualquer modelagem.

---

## PROXIMO

- Gap-filling das 77 datas nubladas (interpolação linear)
- Explorar os dados do Metos Pro para cruzar com as anomalias
- Definir critério de rotulagem de anomalias para o treino da U-Net