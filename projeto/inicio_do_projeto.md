## Contexto

<p align="justify">O monitoramento de pastagens é uma prática essencial para a gestão eficiente da produção animal, permitindo ajustes no manejo e garantindo sustentabilidade. Métodos tradicionais, como análises laboratoriais, apresentam alto custo e baixa agilidade, dificultando sua adoção por pequenos e médios produtores.</p>

<p align="justify">Este projeto propõe uma abordagem prática e acessível, utilizando <b>imagens capturadas por smartphones convencionais</b> para estimar índices de massa foliar. Essa solução busca reduzir custos, aumentar a velocidade de análise e democratizar o acesso a tecnologias de monitoramento no agronegócio.</p>


## Objetivos

### Objetivo Geral
<p align="justify">Realizar uma <b>análise experimental comparativa</b> dos índices de massa foliar estimados por diferentes modelos de aprendizado de máquina, treinados a partir de um conjunto de imagens de pastagens, com o objetivo de avaliar a viabilidade do uso desses modelos como alternativa à análise.</p>

### Objetivos Específicos

- **Processar imagens capturadas por smartphones** e extrair índices de vegetação, como GLI, MPRI, ExR, ExG e ExB.
- **Comparar o desempenho** de diferentes modelos de aprendizado de máquina (Random Forest, Redes Neurais) na inferência da massa foliar.
- **Propor uma abordagem acessível** baseada em ferramentas de código aberto, voltada para pequenos e médios produtores rurais.
- **Desenvolver um pipeline automatizado** para pré-processamento, treinamento e avaliação dos modelos.

### Resultados Esperados

- **Identificar limitações e potencialidades** dos modelos aplicados à estimativa da massa foliar.
- **Desenvolver um fluxo de análise replicável**, utilizando Python e bibliotecas como Scikit-learn, TensorFlow e OpenCV.
- **Contribuir para soluções tecnológicas acessíveis e de baixo custo**, promovendo inovação no monitoramento de pastagens.
- **Gerar métricas comparativas** (R², RMSE, AUC, F1-Score) para fundamentar recomendações sobre o uso dos modelos.
- Contribuir para o **desenvolvimento de soluções tecnológicas acessíveis e de baixo custo** voltadas ao monitoramento de pastagens no contexto do agronegócio.

## Referências

Documentação do Amazon SageMaker.
Artigo disponível em: https://aws.amazon.com/pt/sagemaker/. Acesso em 16/11/2025.

GSENNAURA. AgroBrain Biomass: Sistema Inteligente de Análise de Pastagens. Repositório privado. Disponível em: https://github.com/gsennaura/agrobrain-biomassa/tree/main. Acesso em: 16/11/2025.

EMBRAPA. FAPESP aprova projeto para diagnóstico e monitoramento de pastagens. Portal Embrapa Notícias, 06/06/2025. Disponível em: https://www.embrapa.br/busca-de-noticias/-/noticia/101051991/projeto-em-rede-para-diagnostico-e-monitoramento-de-pastagens-e-aprovado-pela-fapesp. Acesso em: 16/11/2025.

SILVA, M. H. et al. Análise da cultura da soja a partir de índices de vegetação (ExG, GLI, TGI) advindos de imagens RGB obtidas com ARP. *Revista Brasileira de Geomática*, v. 10, n. 2, p. 140-154, 2022. Disponível em: https://periodicos.utfpr.edu.br/rbgeo/article/view/15042. Acesso em: 16/11/2025.






