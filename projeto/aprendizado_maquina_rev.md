## **Ferramentas e Preparação dos Dados para Aprendizagem de Máquina**

Na etapa de **aprendizagem de máquina** do projeto *“Análise experimental comparativa entre os índices de massa foliar obtidos por imagens de satélite e de celular”*, foi utilizada uma base de dados previamente estruturada a partir do pré-processamento das imagens. Esse pré-processamento envolveu:

- **Recorte central das imagens**, visando padronizar a área de análise;
- **Cálculo de estatísticas descritivas** dos pixels, como média, desvio padrão e variância;
- **Separação dos espectros de núcleos** relevantes para a vegetação;
- **Cálculo de índices de massa foliar**, como NDVI e outros índices espectrais.

Os resultados dessas operações foram organizados em uma **tabela estruturada com a biblioteca Pandas**, reunindo todas as estatísticas calculadas por imagem. Essa tabela representa o conjunto de dados “limpos” e prontos para serem utilizados na modelagem preditiva.

Para a etapa de **aprendizado de máquina**, foram explorados dois tipos de algoritmos:

### Redes Neurais Convolucionais (CNN)
Utilizadas para extrair padrões espaciais e visuais diretamente das imagens, as CNNs são especialmente eficazes na análise de dados visuais complexos. Foram aplicadas para identificar características relevantes nas imagens que se correlacionam com os índices de massa foliar.

### Random Forest
Escolhido por sua robustez frente a dados com variáveis correlacionadas e por oferecer boa interpretabilidade dos resultados. O algoritmo foi treinado com os dados tabulados, incluindo as **datas de captura das imagens** e os respectivos **índices de massa foliar**.

Essas ferramentas permitiram a construção de modelos preditivos capazes de estimar os índices de massa foliar com base nas características extraídas das imagens, possibilitando a **comparação entre os resultados obtidos por imagens de satélite e de celular**.

## Aprendizado de Máquina

Inicialmente, o planejamento estratégico previa a execução dos algoritmos em um ambiente cloud de alta performance,
especificamente utilizando o Amazon SageMaker, com o conjunto de dados (imagens) hospedado no Amazon Simple Storage Service (S3).
A integração com o S3 para o upload de todas as imagens foi concluída com sucesso, estabelecendo-o como a fonte central de dados brutos do projeto.

![Image](https://github.com/user-attachments/assets/4ca20391-63c2-46a1-b89f-54b64bfddcf0)

No entanto, a tentativa de operacionalizar a plataforma de ML no cloud encontrou um obstáculo.
Ao prosseguir com a criação do domínio necessário para utilizar o SageMaker, a equipe se deparou com um erro de permissão.
O erro nos diz que a conta "LAB" da AWS, utilizada para o projeto, não concedia as permissões de acesso necessárias para a utilização do serviço Amazon SageMaker.

![Image](https://github.com/user-attachments/assets/2df2b0a9-d2c7-4061-a84c-41f4ef667a19)

Diante dessa restrição imposta pela política da conta, a equipe tomou a decisão de alterar o ambiente de execução. Para garantir o avanço do projeto e a entrega dos resultados de ML,
o treinamento dos algoritmos(Random Forest e Deep Learning) foi feito ambiente local, sendo realizado através do Visual Studio Code. Permitindo que os processos de pré-processamento, treinamento
e avaliação dos modelos fossem realizados de forma eficaz e imediata, mantendo a integridade e cronograma do projeto.
