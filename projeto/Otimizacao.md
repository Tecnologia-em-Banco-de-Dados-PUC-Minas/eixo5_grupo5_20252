# **Otimização**

<p align="justify">A etapa de otimização teve como objetivo confrontar o planejamento de governança de dados estabelecido no início do projeto com as ações efetivamente adotadas ao longo do desenvolvimento. Esse processo permitiu identificar lacunas, validar práticas, aprimorar procedimentos e registrar melhorias aplicadas até o momento — além de apontar direções futuras que podem potencializar o crescimento operacional e científico da solução.</p>

## **6.1 Governança dos Dados: Planejado x Executado**
Durante a segunda etapa do projeto, foi elaborado um plano de governança pautado nos princípios de:
- Prestação de contas
- Transparência
- Conformidade com a legislação (LGPD e GDPR)
- Padronização e qualidade dos dados
- Segurança da informação
- Rastreabilidade

Ao longo do desenvolvimento, esse planejamento foi refletido nas ações adotadas pela equipe:
- Repositório de dados privado e acessível apenas aos pesquisadores autorizados
- Decisões de manipulação, pré-processamento e modelagem documentadas em histórico
- Medidas de proteção contra exposição indevida das informações
- Ações corretivas como recorte padronizado e normalização para garantir consistência

### **Integridade e qualidade dos dados**

Apesar da preservação da integridade, foram identificadas inconsistências operacionais relacionadas a:
- Distância variável da câmera em relação ao solo
- Variação excessiva de luminosidade
- Diferenças entre sensores de câmeras celulares
- Direção de incidência da luz
- Ângulo de inclinação da câmera

Para mitigar esses efeitos, foi adotada padronização retroativa:
- Recorte central uniforme das imagens
- Normalização estatística
- Eliminação de pixels não representativos nas bordas

Essa ação aumentou a confiabilidade dos índices **GLI, MPRI, ExG, ExR e ExB**.

## **6.2 Pontos de Melhoria Identificados**

### **1. Aumento da Base de Dados**
Com maior volume de imagens, é esperado:
- Melhor capacidade de aprendizado da rede neural
- Redução no erro absoluto
- Menor risco de overfitting
- Maior estabilidade da Random Forest

Além disso, com expansão do dataset será possível:
- Segmentar os dados por período do ano
- Comparar estação seca vs. estação chuvosa
- Correlacionar fatores meteorológicos

### **2. Intervalo Temporal dos Dados**
Definição clara do período de coleta das imagens, incluindo:
- Data exata
- Horário da captura
- Local georreferenciado (se possível)
- Informações climatológicas associadas (opcional)

Essa contextualização permitirá análises avançadas como:
- Influência do índice pluviométrico na massa foliar
- Oscilações sazonais
- Taxas de regeneração do pasto

### **3. Padronização da Coleta de Dados**
Proposta de criação de protocolo operacional:
- Uso do mesmo modelo de celular, quando possível
- Altura fixa da câmera em relação ao solo
- Ângulo perpendicular ao plano do capim
- Coleta preferencial entre 10h e 14h
- Desconsiderar dias de sombra profunda ou chuva
- Uso de moldura física (ex.: retângulo de referência no chão)

Além disso, recomenda-se desenvolver um **manual de campo** para coleta fotográfica.

## **6.3 Propostas Futuras e Potencial de Mercado**
A continuidade do projeto abre possibilidades de:
- Expansão gradual do banco de dados com novas imagens
- Ampliação da coleta laboratorial de massa foliar
- Avanço para análises multimodais combinando dados de celular e imagens satelitais

O projeto mantém relevância acadêmica e pode servir como base para:
- Pesquisas de graduação
- Iniciação científica
- Extensão aplicada ao setor agrícola

### **Potencial Comercial**
Em etapa posterior, prevê-se evolução para:
- Assistente digital de manejo sustentável de pastagens
- Inferência preditiva para planejamento nutricional do rebanho
- Estimativa de lotação ideal
- Detecção precoce de degradação da cobertura vegetal

Disponibilização como **serviço SaaS** para:
- Propriedades rurais
- Cooperativas
- Empresas do agronegócio

Inserindo a solução no contexto **AgroTech**, com potencial de:
- Escalabilidade produtiva
- Adoção em larga escala
- Contribuição para uma agricultura mais inteligente, eficiente e sustentável
