# Pré-processamento de Dados  

<p align="justify">O pré-processamento de dados é uma etapa fundamental em qualquer projeto de Machine Learning, pois garante que os dados estejam limpos, padronizados e prontos para serem analisados pelos algoritmos.</p>

<p align="justify">No contexto deste projeto, que busca analisar a nutritividade de um pasto a partir de imagens, o pré-processamento é responsável por reduzir ruídos, organizar informações e extrair estatísticas relevantes, possibilitando maior precisão na etapa de análise e treinamento dos modelos.</p>


## Ferramentas Utilizadas  

O pré-processamento foi realizado inteiramente na linguagem **Python**, utilizando o **Visual Studio Code** como ambiente de desenvolvimento.  

As bibliotecas empregadas foram:  

- **Pandas** → Manipulação, organização e estruturação dos dados em formato tabular (DataFrames), facilitando análises estatísticas e geração de tabelas finais.  
- **NumPy** → Operações matemáticas e manipulação eficiente de arrays, servindo de base para cálculos matriciais e vetoriais.
- **Scikit-image** → Empregada no processamento de imagens, foi utilizada no recorte da imagem e extração das bandas RGB.
- **Matplotlib** → Visualização de dados e gráficos, permitindo observar distribuições, estatísticas e padrões visuais dos índices calculados.  
- **Scikit-learn (sklearn)** → Biblioteca voltada para Machine Learning, utilizada especialmente para apoiar métricas estatísticas e preparação dos dados.  
- **Tensorflow** → Biblioteca principal utilizada para o desenvolvimento e treinamento dos modelos de deep learning, sendo referência na área por sua flexibilidade e desempenho.


## Etapas do Pré-processamento  

### 1. Recorte central da imagem  
<p align="justify">Foi implementada uma função que realiza o <strong>recorte da imagem a partir de seu ponto central</strong>. Esse procedimento teve como objetivo facilitar os cálculos de índices por pixel,reduzindo a possibilidade de ocorrência de <strong>outliers</strong> nas extremidades da imagem, que poderiam distorcer os resultados.</p>

<img width="996" height="94" alt="Image" src="https://github.com/user-attachments/assets/7d92d5e9-9cc6-4f0e-9b58-fa524cc31fd1" />

### 2. Cálculo de estatísticas descritivas  
<p align="justify">Por meio de funções específicas, foram realizados cálculos de estatísticas descritivas como <strong>quartis, média e desvio padrão</strong>. Essas medidas são importantes para compreender a variabilidade dos dados e fornecer uma base sólida para análises posteriores.</p>

<img width="992" height="151" alt="Image" src="https://github.com/user-attachments/assets/ff2064cb-1124-4b83-bf67-60b6c600e175" />

### 3. Separação de espectros de cores e cálculo de índices  

Outra função foi responsável por:    
- Separar os espectros de cores **RGB (Red, Green, Blue)**;  
- Calcular os **5 índices necessários** para a análise;  
- Gerar estatísticas descritivas para cada índice calculado.

<img width="992" height="477" alt="Image" src="https://github.com/user-attachments/assets/c8168ea1-81e8-426e-abc6-083af835c94e" />
<img width="992" height="266" alt="Image" src="https://github.com/user-attachments/assets/cbd2515d-52bf-4d1c-a396-2cf6181aa344" />

<p align="justify">Os resultados obtidos foram organizados em uma <strong>tabela estruturada com a biblioteca Pandas</strong>, reunindo todas as estatísticas calculadas. Essa tabela representa o conjunto de dados “limpos” e prontos para serem utilizados nas próximas etapas de análise e treinamento dos algoritmos de Machine Learning.</p>

