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

## Feedback do orientador — Out/2026

### Ponto 1: Separação treino/teste

Split aleatório não é adequado para série temporal — imagens adjacentes
são quase idênticas, o que infla as métricas artificialmente.

Decisão: adotar **leave-one-field-out (LOFO)** — treinar nos 3 talhões,
testar no 4º. Isso mede se o modelo generaliza para áreas que nunca viu,
que é o que importa numa aplicação real.

### Ponto 2: U-Net de imagem única vs. anomalias temporais

O U-Net não pode detectar "NDVI baixo por 3 datas consecutivas" olhando
só uma imagem — isso é uma inconsistência arquitetural real.

Solução adotada: redefinir a anomalia como **desvio do baseline histórico**,
eliminando a dependência de sequência temporal na inferência:

1. Calcula-se a mediana NDVI por pixel por fase fenológica (usando dados
   de treino)
2. Anomalia = pixel com NDVI atual abaixo do esperado para aquela fase
   (> 1.5 desvio padrão abaixo da mediana histórica)
3. Entrada do U-Net: **2 canais** — imagem NDVI atual + mapa de desvio
   do baseline

Assim o contexto temporal fica "embutido" no segundo canal, e o modelo
opera normalmente com uma única imagem na inferência.

### Impacto no código

- `preprocessamento.py`: adicionar cálculo do baseline por fase fenológica
- `modelo.py`: entrada com 2 canais (in_channels=2) em vez de 1
- `rotulagem.py`: gerar máscaras por desvio do baseline, não por sequência

---

## Decisão: escopo temporal da análise (out/2026)

Ao receber os TIFFs completos, o dataset se revelou maior do que o esperado:
- **428 arquivos** (107 datas × 4 talhões), de ago/2025 até out/2026
- O período original estimado era 196 imagens até fev/2026

Optei por dois cenários paralelos:

**Cenário SAFRA** (`modo="safra"`)
- Período: ago/2025 → mai/2026 (~colheita)
- Foco principal do TCC — cobre o ciclo completo da safra 2025/2026
- Fases: pré-plantio, crescimento, florescimento, colheita

**Cenário COMPLETO** (`modo="completo"`)
- Período: ago/2025 → out/2026
- Inclui pós-colheita e início de possível nova safra
- Usado como dado de embasamento e comparativo

Os dois cenários geram arquivos `.npz` independentes em `data/processed/`.
A análise principal e o modelo U-Net usam o cenário SAFRA.
O cenário COMPLETO entra na discussão dos resultados.
Será analisado posteriormente o comportamento de análise com ambos misturados para estudos de resultados. 

---

## Definição de anomalia — validar com comitê FarmLab

O critério de anomalia adotado no código é:

> **Pixel é anômalo quando seu NDVI está abaixo de 1,5 desvios padrão
> do baseline mediano da sua fase fenológica.**

Em termos práticos:
- O baseline é calculado como a **mediana histórica** de cada pixel,
  separada por fase (pré-plantio / crescimento / florescimento / colheita)
- O desvio é normalizado pelo desvio padrão global do pixel ao longo do tempo
- Limiar: `desvio < -1.5` → anomalia

**Pendente de validação agrônomica:**
- O limiar -1.5σ é conservador ou permissivo demais para o contexto?
- Anomalia inclui apenas estresse hídrico/doença, ou também falha no stand
  (pixel que nunca germinou)?
- Como tratar pixels próximos à borda do talhão (máscara Band 3)?
- Validar visualmente ao menos 2–3 datas com anomalia detectada
  contra as imagens JPG correspondentes

## Arquitetura do modelo (out/2026)

Optei por um **U-Net de 3 níveis** adaptado para imagens pequenas (40×54 px).

**Entrada:** 2 canais
- Canal 0: NDVI atual (gap-filled)
- Canal 1: desvio normalizado do baseline fenológico

**Saída:** máscara binária 40×54 (sigmoid → anomalia)

**Ajuste técnico:** a largura 54 não é divisível por 8 (necessário para
3 níveis de pooling), então o modelo aplica padding interno para 56 e
recorta na saída — transparente para o treino e inferência.

**Função de perda:** BCE + Dice Loss (50/50)
- BCE penaliza cada pixel individualmente
- Dice Loss lida melhor com o desbalanceamento de classes
  (pixels anômalos são minoria nas imagens)

Parâmetros treináveis: ~247 mil — leve o suficiente para treinar em CPU.