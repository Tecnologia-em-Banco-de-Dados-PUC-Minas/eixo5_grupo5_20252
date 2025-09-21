## Coleta de Dados e Governança

### Coleta de Dados

O projeto consiste na análise experimental de pastagens em uma fazenda de criação de gado para corte, com o objetivo de avaliar métodos alternativos para a verificação da massa foliar. O estudo corresponde a uma colaboração com os autores principais, onde o grupo busca contribuir com o desenvolvimento de uma solução alternativa de análise de dados e processamento de informações que poderá servir como validação das técnicas e métodos utilizados no estudo original.

Todos os dados utilizados nesse projeto foram coletados pelos pesquisadores responsáveis pelo estudo original e estão armazenados em um repositório privado no GitHub do projeto denominado **AgroBrain Biomass: Sistema Inteligente de Análise de Pastagens**. O acesso ao repositório original foi franqueado aos integrantes do grupo deste estudo secundário, denominado **Projeto de Arquitetura de Dados em Nuvem: Análise experimental de comparação entre índices de massa foliar obtidos através de imagens de satélite e celular**.

A coleta de dados envolve inicialmente duas fontes principais, que se complementam:

- **Imagens de celular**: Fotografias capturadas em campo nos piquetes de pastagem, utilizando smartphones com câmeras de alta resolução. Cada fotografia corresponde a uma amostra de pastagem que também é analisada em laboratório, garantindo um registro preciso da área estudada.
- **Análises laboratoriais**: Dados da massa foliar real obtidos a partir das amostras coletadas em cada piquete. Esses valores servem como referência para validação e comparação com os índices extraídos das imagens. Os resultados das análises foram reunidos e armazenados no repositório do GitHub.

Ao avançar na pesquisa, é possível que seja incluída uma terceira fonte de dados:

- **Imagens de satélite (Sentinel-2)**: Já se encontram armazenadas no repositório privado do GitHub destinado aos documentos do estudo.

Para garantir a qualidade e consistência dos dados, as imagens de celular são processadas por algoritmos em Python que calculam índices de vegetação por pixel, como:

- GLI
- MPRI
- ExG
- ExR
- ExB

Esses índices são posteriormente convertidos em valores médios por imagem. Cada registro é vinculado ao piquete correspondente, permitindo rastreabilidade completa entre imagem e análise laboratorial.

O **tratamento de dados** deste projeto é executado em uma máquina local dedicada, escolhida pela sua capacidade de processamento para manipular o volume de imagens de alta resolução, superando as limitações de ambientes de nuvem gratuitos como Google Colab. Os dados são **armazenados e versionados** em um repositório do GitHub, são processados por meio de algoritmos em Python desenvolvidos na IDE **Visual Studio Code**.

Utilizamos um conjunto de bibliotecas especializadas para todo o fluxo de tratamento: **NumPy** para calcular os índices de vegetação (como GLI e ExG) pixel a pixel e agregar suas médias, **Pandas** para integrar esses resultados com os dados laboratoriais de massa foliar, criando um conjunto consolidado e **Scikit-learn e Matplotlib** para a análise e modelagem subsequentes. Essa abordagem garante um pipeline robusto, controlado e eficiente, desde a imagem bruta até os dados prontos para análise.

### Governança de Dados

A governança de dados é um componente essencial para assegurar que as informações coletadas sejam confiáveis, rastreáveis, seguras e conformes às regulamentações aplicáveis, como a **LGPD** e a **GDPR**. Além de garantir a proteção de dados pessoais, a governança fornece base para decisões informadas, eficiência operacional e inovação organizacional.

#### Princípios e Relevância

A gestão de dados é fundamental para:

- Garantir dados precisos, confiáveis e atualizados para a tomada de decisões estratégicas;
- Melhorar a eficiência operacional, reduzindo duplicidade de esforços e acelerando o acesso às informações;
- Assegurar qualidade e integridade, através de validação, padronização e aplicação de regras de negócio;
- Atender às exigências de conformidade regulatória, incluindo consentimento e proteção de dados pessoais;
- Proteger os dados contra acessos não autorizados, mantendo confidencialidade e integridade.


#### Requisitos de Governança do Projeto

Considerando a parceria com o estudo original, os principais requisitos definidos são:

- **Segurança e confidencialidade**: Garantir que o acesso à base de dados seja controlado, preservando as informações contra acessos não autorizados e divulgação indevida.
- **Qualidade e padronização das imagens**: Assegurar que todas as imagens de celular coletadas estejam em formatos padronizados e com qualidade adequada para análise.
- **Rastreabilidade**: Registrar a origem de cada dado (data, hora, piquete, GPS), assegurando que seja possível vincular imagens aos resultados laboratoriais correspondentes.
- **Compliance regulatório**: Garantir que os dados tratados estejam em conformidade com LGPD/GDPR, mesmo sendo informações não pessoais diretamente relacionadas a indivíduos.
- **Integração e consistência**: Manter uniformidade entre as diferentes fontes de dados, incluindo imagens de celular, análises laboratoriais e futuras imagens de satélite.


#### Estrutura e Processos de Governança

A governança de dados neste projeto é estruturada para assegurar que todas as etapas do ciclo de vida dos dados — desde a coleta até o armazenamento e processamento — sejam conduzidas de forma segura, organizada e padronizada. Isso inclui:

- Manutenção da integridade e consistência das informações;
- Garantia de rastreabilidade de cada registro até sua fonte original (imagem de celular, análise laboratorial ou imagem de satélite);
- Acesso restrito aos dados, apenas por usuários autorizados;
- Suporte sólido para análises futuras, tomada de decisão informada e desenvolvimento de soluções que contribuam com eficiência, segurança e inovação no monitoramento de pastagens conforme definido no estudo original.





