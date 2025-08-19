## Contexto

Este projeto utilizará dois conjuntos principais de dados:

- **Imagens de Satélite**: obtidas de fontes públicas como o Sentinel-2 (via Copernicus), que oferecem dados espectrais úteis para o cálculo de índices de vegetação como NDVI (Normalized Difference Vegetation Index).  O Índice de Vegetação da Diferença Normalizada), é usado para medir a saúde e a densidade da vegetação através de imagens de satélite.
- **Imagens de Celular**: fotografias capturadas diretamente em campo, com smartphones comuns, representando uma alternativa de baixo custo e alta acessibilidade.

A escolha desses conjuntos se justifica pela complementaridade entre eles: enquanto as imagens de satélite oferecem cobertura ampla e dados espectrais, as imagens de celular permitem análises locais com alta resolução. 

---

## Objetivos

### Objetivo Geral
Realizar uma análise experimental comparativa entre os índices de massa foliar obtidos por imagens de satélite e de celular, visando avaliar a viabilidade do uso combinado dessas fontes para o monitoramento de pastagens.

### Objetivos Específicos
- Processar e extrair índices de vegetação (como NDVI) a partir de imagens de satélite e celular.
- Comparar a precisão e correlação entre os dados obtidos por ambas as fontes.
- Avaliar a influência de fatores como resolução, iluminação e tipo de vegetação nos resultados.
- Propor um modelo de análise acessível para produtores rurais, utilizando ferramentas de código aberto.

### Resultados Esperados
- Identificação de limitações e potencialidades de cada tipo de imagem.
- Desenvolvimento de um fluxo de análise replicável com Python.
- Contribuição para soluções tecnológicas de baixo custo no agronegócio.
