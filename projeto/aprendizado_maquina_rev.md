## **Ferramentas e Preparação dos Dados para Aprendizagem de Máquina**

Na etapa de **aprendizagem de máquina** do projeto *“Análise experimental comparativa entre os índices de massa foliar obtidos por imagens de celular”*, foi utilizada uma base de dados previamente estruturada a partir do pré-processamento das imagens. Esse pré-processamento envolveu:

- **Recorte central das imagens**, visando padronizar a área de análise;
- **Cálculo de índices de massa foliar**, como GLI e outros índices espectrais.
- **Cálculo de estatísticas descritivas** dos índices de vegetação, como média, desvio padrão e percentis;

O objetivo principal foi identificar padrões entre as características extraídas das imagens de pastagens e os valores de massa foliar obtidos por análise laboratorial, a fim de avaliar o potencial do uso combinado de diferentes fontes de imagem na estimativa da qualidade das pastagens.

A base de dados utilizada nesta etapa foi derivada do pré-processamento das imagens descrito na **Etapa 3**, organizada em uma tabela estruturada com a biblioteca **Pandas**.

Essa tabela é constituída por índices espectrais calculados para cada amostra de imagem e está disponível em um arquivo tabular denominado `path_img_by_lab_sample.csv`.

Este arquivo reúne:

- Os caminhos relativos das imagens
- Seus respectivos valores de massa foliar

Esse conjunto de dados funciona como o ponto de partida para o processo de **modelagem**.

Para a etapa de **aprendizado de máquina**, foram explorados três tipos de algoritmos:

### Random Forest
Escolhido pelo seu desempenho e simplicidade frente a dados com variáveis correlacionadas de maneira não linear e por oferecer boa interpretabilidade dos resultados. O algoritmo foi treinado com os dados tabulados, incluindo as **Estatísticas descritivas dos índices de vegetação** e os respectivos **índices de massa foliar**.

O Random Forest é um algoritmo de ensemble baseado em árvores de decisão. Quando se combina várias árvores, temos uma floresta. Ele cria várias árvores de decisão usando conjuntos de dados aleatórios e, em seguida, combina as previsões de cada árvore para produzir uma previsão final. O Random Forest é um conjunto de várias árvores de decisão que trabalham juntas para fazer previsões mais precisas. Ao invés de depender de uma única árvore, ele cria múltiplas árvores e combina suas respostas. Isso o torna mais robusto e menos propenso a erros causados por variações nos dados. Ele usa a votação entre árvores para prever categorias e a média das previsões para problemas de regressão.

Como funciona?

Criação de várias árvores de decisão → O algoritmo constrói várias árvores, cada uma com um conjunto ligeiramente diferente de dados.

Cada árvore faz uma previsão → Quando recebe um novo dado, cada árvore dá um "palpite" sobre a classe correta.

Votação das árvores (Classificação) → No caso de classificação, cada árvore vota e a resposta mais escolhida entre todas é a decisão final.

Média das previsões (Regressão) → Para problemas de regressão, o resultado final é uma média das previsões feitas pelas árvores.

A imagem abaixo apresenta um trecho do código em Python utilizado na construção do modelo de regressão usado. Para avaliar o desempenho dos algoritmos, os dados foram divididos em conjuntos de treinamento e teste (relação 80/20), adotando também validação cruzada para reduzir viés na avaliação.

![Random Forest](https://github.com/Tecnologia-em-Banco-de-Dados-PUC-Minas/eixo5_grupo5_20252/blob/experiment/ml-models-v1-results/projeto/src/random-forest/random-forest.py#L1C1-L419C11)


### Redes Neurais
Escolhido por sua capacidade de se ajustar a padrões complexos dos dados e por sua eficiência no tempo de treinamento. O algoritmo foi treinado com os dados tabulados, incluindo as **Estatísticas descritivas dos índices de vegetação** e os respectivos **índices de massa foliar**.

### Redes Neurais Convolucionais (CNN)
Utilizadas para extrair padrões espaciais e visuais diretamente das imagens, as CNNs são especialmente eficazes na análise de dados visuais complexos. Foram aplicadas para identificar a correlação dos índices de massa foliar com as imagens multibanda criadas com cada banda sendo um dos índices de vegetação.

Essas ferramentas permitiram a construção de modelos preditivos capazes de estimar os índices de massa foliar com base nas características extraídas das imagens.

Essa combinação reforça o caráter exploratório e científico do projeto, buscando explorar qual apresenta o melhor equilíbrio entre desempenho estatístico, estabilidade e aplicabilidade prática no contexto do manejo das pastagens.

## Aprendizado de Máquina

Inicialmente, o planejamento estratégico previa a execução dos algoritmos em um ambiente cloud de alta performance, especificamente utilizando o Amazon SageMaker, com o conjunto de dados (imagens) hospedado no Amazon Simple Storage Service (S3).
A integração com o S3 para o upload de todas as imagens foi concluída com sucesso, estabelecendo-o como a fonte central de dados brutos do projeto.

![Image](https://github.com/user-attachments/assets/4ca20391-63c2-46a1-b89f-54b64bfddcf0)

No entanto, a tentativa de operacionalizar a plataforma de ML no cloud encontrou um obstáculo.
Ao prosseguir com a criação do domínio necessário para utilizar o SageMaker, a equipe se deparou com um erro de permissão.
O erro nos diz que a conta "LAB" da AWS, utilizada para o projeto, não concedia as permissões de acesso necessárias para a utilização do serviço Amazon SageMaker.

![Image](https://github.com/user-attachments/assets/2df2b0a9-d2c7-4061-a84c-41f4ef667a19)

Diante dessa restrição imposta pela política da conta, a equipe tomou a decisão de alterar o ambiente de execução. Para garantir o avanço do projeto e a entrega dos resultados de ML, o treinamento dos algoritmos (Random Forest e Deep Learning) foi feito ambiente local, sendo realizado através do Visual Studio Code. Permitindo que os processos de pré-processamento, treinamento e avaliação dos modelos fossem realizados de forma eficaz e imediata, mantendo a integridade e cronograma do projeto.

## Desafios e Estratégias de Superação

Durante o desenvolvimento desta etapa, a equipe enfrentou diversos desafios técnicos e operacionais que influenciaram a execução dos experimentos. Entre os principais, destacam-se:

### 1. Limitações do ambiente cloud
A conta educacional da AWS apresentou restrições de permissão que impossibilitaram o uso do serviço **Amazon SageMaker**, exigindo a migração para execução local.  
**Solução adotada:** uso do **Visual Studio Code** com bibliotecas Python equivalentes, garantindo continuidade ao projeto sem comprometer a integridade metodológica.

### 2. Heterogeneidade das imagens
As fotografias de pastagens foram obtidas por diferentes dispositivos e operadores, com variações em iluminação, distância, ângulo e resolução.  
**Estratégia:** recorte central padronizado das imagens, buscando isolar a região mais representativa da vegetação.

### 3. Dispersão temporal das coletas
As imagens foram capturadas em diferentes datas, sob condições ambientais variáveis.  
**Decisão:** manter a diversidade temporal para aumentar a robustez dos modelos, mesmo com o ruído introduzido.

### 4. Tamanho reduzido do banco de dados
O conjunto de imagens disponível ainda é limitado, o que afeta a capacidade preditiva dos modelos mais complexos.  
**Perspectiva:** como se trata de um projeto colaborativo e em andamento, prevê-se a expansão contínua da base de dados, permitindo reavaliações e aperfeiçoamentos progressivos dos modelos de aprendizado de máquina.

Essas ações demonstram a adoção de boas práticas de **engenharia de dados** e **experimentação**, preservando a consistência do estudo mesmo diante das limitações práticas do ambiente e das fontes de dados.

## Considerações Finais

A implementação das abordagens de **aprendizado de máquina** representou um avanço significativo no projeto, permitindo a integração entre dados visuais e laboratoriais em um fluxo analítico coerente.

Apesar das dificuldades técnicas e das limitações da base de dados, foi possível estruturar modelos capazes de gerar **estimativas iniciais da massa foliar** com base nas características espectrais das imagens.

Os resultados parciais obtidos nesta etapa serão aprofundados nas etapas seguintes, nas quais serão apresentadas:

- Métricas de desempenho
- Comparações entre os modelos
- Interpretações sobre a influência das variáveis na precisão das predições






