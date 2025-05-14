# Assistente de Consulta de Normas Técnicas (RAG)

## Autor

- [*Gustavo Rodrigues Ribeiro*](https://github.com/GustavooRibas)

Este projeto implementa um assistente virtual baseado em RAG (Retrieval-Augmented Generation) para responder perguntas sobre normas técnicas específicas. Ele utiliza o framework LangChain para orquestrar o processo de busca em documentos e geração de respostas.

## Funcionalidades

*   Indexa documentos PDF contendo normas técnicas.
*   Permite consultas em linguagem natural sobre o conteúdo das normas.
*   Cita o nome do arquivo da norma de onde a informação foi extraída.
*   Responde "Não encontrei essa informação nas normas disponíveis" para consultas fora do escopo.

## Pré-requisitos

*   Python 3.8 ou superior
*   pip (gerenciador de pacotes Python)
*   Acesso à API da OpenAI (ou configuração para usar um LLM/Embedding local)

## Como Executar o Projeto

Siga estas etapas para configurar e executar o projeto localmente:

### 1. **Obtenha o Código Fonte:**

```bash
# Exemplo usando git (se aplicável)
git clone <url-do-repositorio>
cd assistente_normas
```

Ou baixe os arquivos (`assistant.py`, `requirements.txt`, etc.) e coloque-os em uma pasta de projeto (ex: `assistente_normas`).

### 2. **Pastas Necessárias:**

Dentro da pasta do projeto (`assistente_normas`), teremos as seguintes subpastas:

*   `docs`: Esta pasta armazenará os arquivos PDF das normas técnicas. (O arquivo do projeto já possui, em formato PDF, as normas IEEE 802.15.3-2023, IEEE 7002-2022, IEEE 7001-2021, IEEE 1801-2024, IEEE 1685-2022 e IEEE 1666-2023)
*   `vectorstore`: Esta pasta armazenará o índice vetorial criado pelo FAISS. (Pasta vazia até a execução do Assistente)

### 3. **Crie um ambiente virtual (opcional, mas recomendado)**

*    No repositório do projeto:

Criando ambiente virtual:

```bash
python -m venv venv
```

*    Acessando ambiente virtual:

Em Sistemas Unix:

```bash
source venv/bin/activate
```

Em Sistemas Windows (CMD):

```bash
venv\Scripts\activate
```

### 4. **Instale as Dependências:**

Utilize o arquivo `requirements.txt` incluso:

```bash
pip install -r requirements.txt
```

*Observação:* Se você optou por `faiss-gpu` no `requirements.txt`, certifique-se de ter os drivers NVIDIA e o CUDA Toolkit compatíveis instalados.

### 5. **Obtenha as Normas:**

*   Faça o download dos arquivos PDF das normas técnicas que você deseja consultar (por exemplo, as normas IEEE listadas ou normas ABNT do site do Inmetro).
*   **Coloque os arquivos PDF baixados diretamente dentro da pasta `docs`.**
*   **OBS: O arquivo disponível já possui as normas da IEEE, citadas anteriormente, dentro da pasta `docs`.**
*   **Substitua as normas já existentes se busca utilizar outras normas.**

### 6. **Configure a Chave da API OpenAI:**

*   Crie um arquivo chamado `.env` na pasta raiz do projeto (`assistente_normas`).
*   Abra o arquivo `.env` e adicione sua chave da API OpenAI no seguinte formato:
```
OPENAI_API_KEY="sua_chave_api_aqui"
```
*   Salve o arquivo. O script `assistant.py` usará esta chave para acessar os modelos da OpenAI (Embeddings e LLM). Se você não tem uma chave, precisará criar uma conta na plataforma da OpenAI.

## Executando o Assistente

1.  **Execute o Script Python:**
    Abra seu terminal, navegue até a pasta raiz do projeto (`assistente_normas`) e execute o script:
    ```bash
    python assistant.py
    ```

2.  **Primeira Execução (Indexação):**
    Na primeira vez que você executar o script *com novos documentos na pasta `docs`*, ele realizará as seguintes etapas:
    *   Carregará os PDFs da pasta `docs`.
    *   Dividirá o texto em chunks menores.
    *   Gerará embeddings (vetores) para cada chunk usando a API da OpenAI.
    *   Criará o índice vetorial FAISS e o salvará na pasta `vectorstore`.
    *   **Este processo pode levar algum tempo**, dependendo do número e tamanho dos PDFs e da velocidade da sua conexão/API. O script exibirá logs indicando o progresso.

3.  **Execuções Subsequentes:**
    Se o índice FAISS já existir na pasta `vectorstore` (de uma execução anterior), o script o carregará rapidamente, pulando a etapa de processamento dos PDFs e criação do índice. A inicialização será muito mais rápida.

4.  **Interagindo com o Assistente:**
    *   Após a inicialização, o assistente solicitará que você digite sua pergunta:
        ```
        Sua pergunta:
        ```
    *   Digite sua pergunta em linguagem natural sobre as normas carregadas e pressione Enter.
    *   O assistente processará a pergunta, buscará informações relevantes nos documentos indexados e gerará uma resposta usando o LLM.
    *   A resposta incluirá a citação da(s) fonte(s) (Nome do arquivo PDF e Seção) se a informação for encontrada.
    *   Se a informação não for encontrada, ele responderá: "Não encontrei essa informação nas normas disponíveis."
    *   Para sair do assistente, digite `sair` e pressione Enter.

## Estrutura do Projeto

```
assistente_normas/
│
├── docs/ # Coloque os arquivos PDF das normas aqui (o arquivo do projeto já possui 6 normas IEEE)
│ ├── Norma_IEEE_XXXX-YYYY.pdf
│ └── ...
│
├── vectorstore/ # O índice FAISS será salvo aqui
│ └── faiss_index/
│ ├── index.faiss
│ └── index.pkl
│
├── .env # Arquivo para a chave da API
├── assistant.py # O código principal do assistente RAG
├── requirements.txt # Lista de dependências Python
├── README.md # Este arquivo de instruções
└── exemplos_consultas_ieee.txt # Exemplos de perguntas e respostas
```

## Observações

*   **Custo da API:** O uso dos modelos da OpenAI (embedding e LLM) consome créditos da API, que podem ter custos associados. Monitore seu uso na plataforma da OpenAI.
*   **Qualidade do PDF:** A qualidade da extração de texto depende da qualidade dos PDFs. PDFs baseados em imagem ou com formatação complexa podem não ser processados corretamente pelo `PyPDFLoader`.
*   **Precisão:** A precisão da resposta depende da qualidade dos embeddings, da relevância dos chunks recuperados e da capacidade do LLM de seguir as instruções do prompt.

## Contato

- E-mail: gustavooriibeiro.ofc@hotmail.com
