# Pré-processamento de Dados  

O pré-processamento de dados é uma etapa fundamental em qualquer projeto de Machine Learning, pois garante que os dados estejam limpos, padronizados e prontos para serem analisados pelos algoritmos.  

No contexto deste projeto, que busca analisar a nutritividade de um pasto a partir de imagens, o pré-processamento é responsável por reduzir ruídos, organizar informações e extrair estatísticas relevantes,
possibilitando maior precisão na etapa de análise e treinamento dos modelos.  

---

## Ferramentas Utilizadas  

O pré-processamento foi realizado inteiramente na linguagem **Python**, utilizando o **Visual Studio Code** como ambiente de desenvolvimento.  

As bibliotecas empregadas foram:  

- **Pandas** → Manipulação, organização e estruturação dos dados em formato tabular (DataFrames), facilitando análises estatísticas e geração de tabelas finais.  
- **NumPy** → Operações matemáticas e manipulação eficiente de arrays, servindo de base para cálculos matriciais e vetoriais.  
- **Matplotlib** → Visualização de dados e gráficos, permitindo observar distribuições, estatísticas e padrões visuais dos índices calculados.  
- **Scikit-learn (sklearn)** → Biblioteca voltada para Machine Learning, utilizada especialmente para apoiar métricas estatísticas e preparação dos dados.  

---

## Etapas do Pré-processamento  

### 1. Recorte central da imagem  
Foi implementada uma função que realiza o **recorte da imagem a partir de seu ponto central**. Esse procedimento teve como objetivo facilitar os cálculos de índices por pixel,
reduzindo a possibilidade de ocorrência de **outliers** nas extremidades da imagem, que poderiam distorcer os resultados.  

### 2. Cálculo de estatísticas descritivas  
Por meio de funções específicas, foram realizados cálculos de estatísticas descritivas como **quartis, média e desvio padrão**. Essas medidas são importantes para compreender
a variabilidade dos dados e fornecer uma base sólida para análises posteriores.  

### 3. Separação de espectros de cores e cálculo de índices  
Outra função foi responsável por:  
- Realizar um **novo recorte central da imagem**;  
- Separar os espectros de cores **RGB (Red, Green, Blue)**;  
- Calcular os **cinco índices necessários** para a análise de nutritividade;  
- Gerar estatísticas descritivas para cada índice calculado.  

### 4. Organização dos dados em tabela  
Os resultados obtidos foram organizados em uma **tabela estruturada com a biblioteca Pandas**, reunindo todas as estatísticas calculadas. Essa tabela representa o conjunto de dados
“limpos” e prontos para serem utilizados nas próximas etapas de análise e treinamento dos algoritmos de Machine Learning.  

---
