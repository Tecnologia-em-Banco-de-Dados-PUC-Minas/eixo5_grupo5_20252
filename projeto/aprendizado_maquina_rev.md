# Aprendizado de Maquina

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
