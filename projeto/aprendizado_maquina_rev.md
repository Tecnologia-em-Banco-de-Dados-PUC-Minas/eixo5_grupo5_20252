# Ferramentas e Preparação dos Dados para Aprendizagem de Máquina

<p align="justify">Na etapa de <strong>aprendizagem de máquina</strong> do projeto <i>“Análise experimental comparativa entre os índices de massa foliar obtidos por imagens de celular”</i>, foi utilizada uma base de dados previamente estruturada a partir do pré-processamento das imagens. Esse pré-processamento envolveu:</p>

- **Recorte central das imagens**, visando padronizar a área de análise;
- **Cálculo de índices de massa foliar**, como GLI e outros índices espectrais.
- **Cálculo de estatísticas descritivas** dos índices de vegetação, como média, desvio padrão e percentis;

<p align="justify">O objetivo principal foi identificar padrões entre as características extraídas das imagens de pastagens e os valores de massa foliar obtidos por análise laboratorial, a fim de avaliar o potencial do uso combinado de diferentes fontes de imagem na estimativa da qualidade das pastagens.</p>

<p align="justify">A base de dados utilizada nesta etapa foi derivada do pré-processamento das imagens descrito na **Etapa 3**, organizada em uma tabela estruturada com a biblioteca <b>Pandas</b>.</p>

Essa tabela é constituída por índices espectrais calculados para cada amostra de imagem e está disponível em um arquivo tabular denominado `path_img_by_lab_sample.csv`.

Este arquivo reúne:

- Os caminhos relativos das imagens
- Seus respectivos valores de massa foliar

Esse conjunto de dados funciona como o ponto de partida para o processo de **modelagem**.

Para a etapa de **aprendizado de máquina**, foram explorados três tipos de algoritmos:

## Random Forest
Escolhido pelo seu desempenho e simplicidade frente a dados com variáveis correlacionadas de maneira não linear e por oferecer boa interpretabilidade dos resultados. O algoritmo foi treinado com os dados tabulados, incluindo as **Estatísticas descritivas dos índices de vegetação** e os respectivos **índices de massa foliar**.

O algoritmo Random Forest é um método de aprendizado supervisionado baseado em múltiplas árvores de decisão, utilizado para tarefas de classificação e regressão. Sua robustez decorre da combinação de várias árvores, cada uma treinada com subconjuntos aleatórios de dados e atributos, reduzindo o risco de overfitting e aumentando a generalização.

Como funciona?

Criação de várias árvores de decisão → O algoritmo constrói várias árvores, cada uma com um conjunto ligeiramente diferente de dados.

Cada árvore faz uma previsão → Quando recebe um novo dado, cada árvore dá um "palpite" sobre a classe correta.

Votação das árvores (Classificação) → No caso de classificação, cada árvore vota e a resposta mais escolhida entre todas é a decisão final.

Média das previsões (Regressão) → Para problemas de regressão, o resultado final é uma média das previsões feitas pelas árvores.

No contexto do projeto Arquitetura de Dados em Nuvem: Análise experimental de comparação entre índices de massa foliar obtidos através de imagens de celular, o Random Forest foi aplicado tanto para regressão quanto para classificação, utilizando dados tabulados derivados das imagens e estatísticas descritivas dos índices de vegetação:


### Regressão
O modelo apresentou R² = 0.571 no treino e R² = 0.375 no teste, com RMSE de 1103 e 1444, respectivamente. Esses resultados indicam que o algoritmo captura parte da variabilidade dos índices de massa foliar, mas há dispersão significativa em valores altos, sugerindo necessidade de ajustes ou inclusão de variáveis complementares.

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/experiment/ml-models-v1-results/projeto/src/metrics/random-forest/graphic_regression_scatter_result.png)

**Interpretação:**
O modelo apresenta desempenho moderado na regressão, com tendência a subestimar valores altos. A dispersão indica que o Random Forest captura parte da variabilidade, mas há espaço para ajustes (ex.: tuning de hiperparâmetros ou inclusão de mais variáveis explicativas).

### Classificação Binária
A curva ROC revelou excelente desempenho, com AUC = 0.939 no treino e 0.859 no teste, e acurácia superior a 80%. Isso demonstra alta capacidade de discriminação entre classes, mesmo em cenários com dados heterogêneos.

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/experiment/ml-models-v1-results/projeto/src/metrics/random-forest/graphic_binary_classifition_roc_curve_result.png)

**Interpretação:**
Excelente capacidade discriminativa, especialmente no treino. No teste, AUC > 0.85 indica bom desempenho geral, com leve redução, sugerindo generalização adequada.

### Classificação Multiclasse
O modelo manteve consistência, com acurácia de 82% e F1 médio de 0.712 no teste. As curvas ROC para cada classe apresentaram AUC acima de 0.88, reforçando a robustez do algoritmo para diferentes categorias de índices.

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/experiment/ml-models-v1-results/projeto/src/metrics/random-forest/graphic_multiclass_classification_roc_curve_result.png)

**Interpretação:**
O modelo mantém alta capacidade de separação entre classes, com métricas consistentes entre treino e teste. O F1 indica equilíbrio entre precisão e recall.

Esses resultados confirmam que o Random Forest é adequado para lidar com dados complexos e não lineares, comuns em imagens agrícolas capturadas por dispositivos móveis, e permite identificar variáveis mais relevantes para a estimativa da massa foliar, fornecendo insights valiosos para práticas de manejo.

A imagem abaixo apresenta o trecho do código em Python utilizado na construção do modelo de regressão usado. Para avaliar o desempenho dos algoritmos, os dados foram divididos em conjuntos de treinamento e teste (relação 80/20), adotando também validação cruzada para reduzir viés na avaliação.

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/main/projeto/images/Random%20Forest%201.png)


## Redes Neurais
As **Redes Neurais** foram escolhidas pela sua capacidade de se ajustar a padrões complexos nos dados e pela eficiência no tempo de treinamento. O algoritmo foi treinado com dados tabulados, incluindo estatísticas descritivas dos índices de vegetação e os respectivos índices de massa foliar.

### Redes Neurais Convolucionais (CNN)
As **CNNs** foram utilizadas para extrair padrões espaciais e visuais diretamente das imagens, sendo especialmente eficazes na análise de dados visuais complexos. No projeto, as CNNs foram aplicadas para identificar a correlação dos índices de massa foliar com as imagens multibanda, onde cada banda representa um índice de vegetação.

**Como funcionam as Redes Neurais?**

* Camadas e Neurônios:
Cada rede é composta por camadas de neurônios que processam os dados em diferentes níveis de abstração.

* Ajuste de Pesos:
Durante o treinamento, os pesos das conexões são ajustados para minimizar o erro entre a previsão e o valor real.

* CNNs:
Utilizam filtros convolucionais para capturar padrões locais nas imagens, como texturas e variações de cor, fundamentais para estimar a massa foliar.

#### Influência do Número de Neurônios
!Gráfico R² vs Número médio de neurônios

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/experiment/ml-models-v1-results/projeto/src/metrics/neural-network/graphic_regression_scatter_r2_mean_neu.png)

**Observação:**
O R² variou entre aproximadamente 0.47 e 0.65, com tendência de concentração entre 0.58 e 0.62 para redes com número médio de neurônios entre 400 e 700.

**Interpretação:**
Redes muito pequenas ou muito grandes não apresentaram ganhos significativos, indicando que existe um ponto ótimo de complexidade.

#### Influência do Número de Camadas
!Gráfico R² vs Número de camadas

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/experiment/ml-models-v1-results/projeto/src/metrics/neural-network/graphic_regression_scatter_r2_layers.png)

**Observação:**
O R² também variou entre 0.47 e 0.65, com melhor desempenho em redes com 4 a 6 camadas, sugerindo que profundidade moderada é mais eficaz.

**Interpretação:**
Redes muito rasas ou muito profundas tendem a apresentar menor estabilidade, reforçando a importância do ajuste arquitetural.

As Redes Neurais, especialmente as CNNs, mostraram-se adequadas para lidar com padrões complexos presentes nas imagens de celular. Os resultados indicam que:

* A arquitetura da rede (número de camadas e neurônios) influencia diretamente o desempenho.
* O melhor equilíbrio foi obtido com redes de complexidade intermediária.
* A abordagem baseada em CNNs permite capturar características visuais relevantes para estimar a massa foliar, fornecendo uma solução robusta para análise agrícola.

A imagem abaixo apresenta o trecho do código em Python utilizado na construção do modelo de redes neurais.

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/main/projeto/images/Neural%20network%201.png)

Essas ferramentas permitiram a construção de modelos preditivos capazes de estimar os índices de massa foliar com base nas características extraídas das imagens.

Essa combinação reforça o caráter exploratório e científico do projeto, buscando explorar qual apresenta o melhor equilíbrio entre desempenho estatístico, estabilidade e aplicabilidade prática no contexto do manejo das pastagens.

## Avaliação dos Modelos Criados, Métricas Utilizadas e Discussão dos Resultados Obtidos

A avaliação dos modelos foi realizada considerando diferentes abordagens (Random Forest e Redes Neurais) aplicadas aos dados derivados das imagens de celular. Foram utilizadas métricas específicas para **regressão e classificação**, permitindo analisar a capacidade preditiva e discriminativa dos algoritmos.

**Métricas Utilizadas**

**Para Regressão:**

* R² (Coeficiente de Determinação): Mede a proporção da variabilidade explicada pelo modelo.
* RMSE (Root Mean Squared Error): Avalia o erro médio quadrático, penalizando grandes desvios.
* MAE (Mean Absolute Error): Indica o erro médio absoluto, útil para interpretar desvios médios.

**Para Classificação:**

* AUC (Área sob a Curva ROC): Mede a capacidade do modelo em distinguir classes.
* Acurácia: Percentual de previsões corretas.
* F1-Score: Equilíbrio entre precisão e recall, especialmente relevante em classes desbalanceadas.

**Resultados Obtidos**

**Random Forest**

**Regressão:**

* Treino: R² = 0.571 | RMSE = 1103 | MAE = 720
* Teste: R² = 0.375 | RMSE = 1444 | MAE = 931

O modelo apresentou desempenho moderado, capturando parte da variabilidade dos índices de massa foliar, mas com dispersão significativa em valores altos.

**Classificação Binária:**

* Treino: AUC = 0.939 | Acurácia = 87.4%
* Teste: AUC = 0.859 | Acurácia = 81.6%

Excelente capacidade discriminativa, com AUC acima de 0.85 no teste, indicando robustez.

**Classificação Multiclasse:**

* Teste: Acurácia = 82.1% | F1 = 0.712
* AUC por classe: Classe 0 = 0.92 | Classe 1 = 0.89 | Classe 2 = 0.88

O modelo manteve consistência entre treino e teste, com bom equilíbrio entre precisão e recall.

**Redes Neurais**

**Influência da Arquitetura:**

* R² variou entre 0.47 e 0.65, com melhor desempenho em redes com **4 a 6 camadas** e número médio de neurônios entre 400 e 700.

* Redes muito rasas ou muito profundas não apresentaram ganhos significativos.

As Redes Neurais mostraram maior capacidade de ajuste a padrões complexos, mas exigem cuidado na definição da arquitetura para evitar sobreajuste ou subajuste. CNNs foram eficazes na extração de padrões visuais, reforçando sua aplicabilidade em imagens agrícolas.

**Comparação Geral**

* **Random Forest**: Mais interpretável, bom desempenho em classificação, mas limitado na regressão.
* **Redes Neurais**: Melhor captura de padrões complexos, especialmente com CNNs, porém maior custo computacional e necessidade de ajuste fino.

**Conclusão**

Ambos os modelos apresentaram resultados satisfatórios, mas com características distintas:

* **Random Forest** é indicado para cenários que exigem interpretabilidade e menor complexidade computacional.
* **Redes Neurais** são recomendadas para análises visuais mais sofisticadas, aproveitando sua capacidade de aprender padrões complexos.

Ambos os algoritmos foram capazes de extrair padrões relevantes, porém a limitação do conjunto de dados reduziu o potencial máximo de acurácia da análise, influenciando significativamente a performance dos modelos.

À medida que novas imagens forem incorporadas ao repositório original, espera-se que ambos os algoritmos — especialmente a RNA — possam atingir níveis superiores de precisão e robustez.

## Comparação Geral entre os Modelos

| Critério                         | Random Forest                                           | Redes Neurais (MLP/CNN)                               |
|---------------------------------|---------------------------------------------------------|-------------------------------------------------------|
| **Tipo de Dados**               | Tabulados (estatísticas dos índices)                   | Tabulados e imagens multibanda                       |
| **Complexidade Computacional**  | Baixa a moderada                                       | Alta (especialmente CNNs)                            |
| **Interpretabilidade**          | Alta (importância das variáveis)                       | Baixa (modelo caixa-preta)                           |
| **Desempenho em Regressão**     | R² = 0.375 (teste), RMSE = 1444                        | R² entre 0.47 e 0.65 (dependendo da arquitetura)     |
| **Desempenho em Classificação Binária** | AUC = 0.859, Acc = 81.6%                          | Não aplicado diretamente (CNN para padrões visuais)  |
| **Desempenho em Multiclasse**   | Acc = 82.1%, F1 = 0.712                                 | Aplicável via CNN, bom para padrões complexos        |
| **Sensibilidade à Arquitetura** | Baixa (poucos hiperparâmetros críticos)                | Alta (número de camadas e neurônios impacta R²)      |
| **Robustez contra Overfitting** | Alta (ensemble reduz variância)                        | Moderada (necessário regularização e ajuste fino)    |
| **Tempo de Treinamento**        | Rápido                                                 | Mais demorado (especialmente com imagens)            |
| **Aplicabilidade**              | Ideal para dados estruturados e interpretabilidade     | Ideal para padrões visuais complexos                 |

# Aprendizado de Máquina

## Planejamento Inicial e Ajustes Necessários

O planejamento estratégico do projeto previa a execução dos algoritmos em um ambiente **cloud** de alta performance, utilizando o **Amazon SageMaker** com o conjunto de dados (imagens) hospedado no **Amazon Simple Storage Service (S3)**. A integração com o S3 foi concluída com sucesso, estabelecendo-o como a fonte central de dados brutos do projeto.

![Image](https://github.com/user-attachments/assets/4ca20391-63c2-46a1-b89f-54b64bfddcf0)

No entanto, a tentativa de operacionalizar a plataforma de ML na nuvem encontrou um obstáculo: ao criar o domínio necessário para utilizar o SageMaker, a equipe se deparou com um erro de permissão. A conta educacional **AWS LAB**, fornecida pelo convênio com a PUC Minas, não concedia as permissões necessárias para o uso do serviço Amazon SageMaker, conforme imagem abaixo.

![Image](https://github.com/user-attachments/assets/2df2b0a9-d2c7-4061-a84c-41f4ef667a19)

Diante dessa restrição, a equipe decidiu alterar o ambiente de execução. Para garantir o avanço do projeto e a entrega dos resultados, o treinamento dos algoritmos (**Random Forest** e **Redes Neurais**) foi realizado em ambiente **local**, utilizando **Visual Studio Code** e bibliotecas Python equivalentes. Essa decisão permitiu que os processos de pré-processamento, treinamento e avaliação fossem conduzidos de forma eficaz, preservando a integridade metodológica e o cronograma do projeto.

## Desafios e Estratégias de Superação

Durante o desenvolvimento, foram enfrentados diversos desafios técnicos e operacionais:

### 1. Limitações do Ambiente Cloud
- **Problema:** Restrições de permissão na conta educacional da AWS impediram o uso do SageMaker.
- **Solução:** Migração para execução local com Visual Studio Code e bibliotecas Python (Scikit-learn, TensorFlow, Keras).

### 2. Heterogeneidade das Imagens
- **Problema:** Fotografias obtidas por diferentes dispositivos e operadores, com variações de iluminação, ângulo e resolução.
- **Estratégia:** Recorte central padronizado das imagens para isolar a região mais representativa da vegetação.

### 3. Dispersão Temporal das Coletas
- **Problema:** Imagens capturadas em diferentes datas e condições ambientais.
- **Decisão:** Manter diversidade temporal para aumentar a robustez dos modelos, mesmo com ruído adicional.

### 4. Tamanho Reduzido do Banco de Dados
- **Problema:** Conjunto de imagens limitado, afetando a capacidade preditiva de modelos complexos.
- **Perspectiva:** Expansão contínua da base de dados em etapas futuras para reavaliação e aperfeiçoamento dos modelos.

## Considerações sobre Infraestrutura

Outro fator relevante foi a limitação dos computadores locais, que não possuem alta capacidade de processamento para suportar treinamentos extensivos de redes neurais profundas. Essa restrição impactou:
- **Tempo de treinamento:** Mais longo do que o previsto.
- **Complexidade dos modelos:** Necessidade de reduzir número de camadas e neurônios para viabilizar execução.

Apesar dessas limitações, foram adotadas estratégias para otimizar o desempenho:
- Ajuste de hiperparâmetros para reduzir custo computacional.
- Uso de arquiteturas mais leves para Redes Neurais.
- Priorização de modelos interpretáveis e eficientes, como Random Forest.

## Considerações Finais

A implementação das abordagens de aprendizado de máquina representou um avanço significativo no projeto, permitindo integrar dados visuais e laboratoriais em um fluxo analítico coerente.

Mesmo diante das dificuldades técnicas e da limitação da base de dados, foi possível estruturar modelos capazes de gerar estimativas iniciais da massa foliar com base nas características espectrais das imagens.

Os resultados parciais obtidos nesta etapa serão aprofundados nas próximas fases, incluindo:
- **Métricas de desempenho detalhadas**
- **Comparações entre os modelos**
- **Interpretações sobre a influência das variáveis na precisão das predições**

