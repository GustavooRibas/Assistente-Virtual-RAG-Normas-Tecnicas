import os
import logging
from dotenv import load_dotenv

# Componentes do LangChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ============================================
# Assistente Virtual Especializado para Consulta de Normas Técnicas Utilizando a Estrutura RAG
# ============================================
# Autor: Gustavo Rodrigues Ribeiro
# Descrição: Esta aplicação implementa um assistente virtual baseado em RAG
# (Retrieval-Augmented Generation) para responder perguntas sobre normas técnicas 
# específicas. Ele utiliza o framework LangChain para orquestrar o processo de busca 
# em documentos e geração de respostas.
# ============================================

# Setup básico de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar variáveis de ambiente (API Key da OpenAI)
load_dotenv()

# --- Configurações ---
DOCS_PATH = "docs"  # Pasta onde estão os PDFs das normas
VECTORSTORE_PATH = "vectorstore/faiss_index" # Onde salvar/carregar o índice vetorial
MODEL_NAME = "gpt-4" # Modelo LLM da OpenAI
EMBEDDING_MODEL_NAME = "text-embedding-ada-002" # Modelo de embedding da OpenAI (bom e custo-efetivo)
# Alternativa open-source (requer sentence-transformers):
# from langchain_community.embeddings import HuggingFaceEmbeddings
# EMBEDDING_MODEL_NAME_HF = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 1500  # Tamanho dos chunks de texto
CHUNK_OVERLAP = 200 # Sobreposição entre chunks

# --- Funções Auxiliares ---

def load_documents(directory_path: str) -> list:
    """
    Carrega os documentos PDF de um diretório especificado.

    Args:
        directory_path: Caminho para a pasta contendo os arquivos PDF.

    Returns:
        Uma lista de documentos carregados (objetos Document do LangChain).
        Retorna lista vazia se o diretório não existir ou estiver vazio.
    """
    if not os.path.isdir(directory_path):
        logging.error(f"Diretório de documentos não encontrado: {directory_path}")
        return []
    if not os.listdir(directory_path):
         logging.warning(f"Diretório de documentos está vazio: {directory_path}")
         return []

    logging.info(f"Carregando documentos PDF do diretório: {directory_path}")
    # Usar DirectoryLoader para carregar todos os PDFs na pasta
    # glob="**/*.pdf" busca recursivamente, "*.pdf" apenas no nível principal
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    try:
        documents = loader.load()
        logging.info(f"Total de {len(documents)} páginas/documentos carregados.")
        if not documents:
             logging.warning("Nenhum documento foi carregado. Verifique os arquivos PDF no diretório.")
        return documents
    except Exception as e:
        logging.error(f"Erro ao carregar documentos: {e}", exc_info=True)
        return []

def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Divide os documentos carregados em chunks menores.

    Args:
        documents: Lista de documentos carregados.
        chunk_size: Tamanho máximo de cada chunk (em caracteres).
        chunk_overlap: Número de caracteres de sobreposição entre chunks adjacentes.

    Returns:
        Uma lista de chunks de texto (objetos Document do LangChain).
    """
    if not documents:
        logging.warning("Nenhum documento para dividir.")
        return []

    logging.info(f"Dividindo documentos em chunks (tamanho={chunk_size}, sobreposição={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Tenta manter parágrafos/sentenças juntos
        separators=["\n\n", "\n", ".", ",", " ", ""],
        add_start_index=True # Útil para referenciar origem, embora usemos metadados
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Documentos divididos em {len(split_docs)} chunks.")
    return split_docs

def create_or_load_vectorstore(split_docs: list, embeddings, store_path: str) -> FAISS:
    """
    Cria um novo vector store FAISS ou carrega um existente.

    Args:
        split_docs: Lista de chunks de texto para indexar (se for criar um novo).
        embeddings: O modelo de embedding a ser usado.
        store_path: Caminho para salvar/carregar o índice FAISS.

    Returns:
        O objeto FAISS vector store.
    """
    # Verifica se o diretório do vectorstore existe, senão cria
    store_dir = os.path.dirname(store_path)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
        logging.info(f"Diretório do vectorstore criado: {store_dir}")

    if os.path.exists(store_path):
        logging.info(f"Carregando vector store existente de: {store_path}")
        try:
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True) # Necessário para FAISS local
            logging.info("Vector store carregado com sucesso.")
            return vectorstore
        except Exception as e:
            logging.error(f"Erro ao carregar vector store existente: {e}. Criando um novo.", exc_info=True)
            # Se falhar ao carregar, remove o índice antigo e cria um novo
            if os.path.exists(store_path):
                 import shutil
                 shutil.rmtree(store_path) # Remove a pasta do índice corrompido/inválido

    # Se não existir ou falhou ao carregar, cria um novo
    if not split_docs:
         logging.error("Não há documentos divididos para criar um novo vector store.")
         raise ValueError("Não é possível criar um vector store sem documentos.")

    logging.info("Criando um novo vector store...")
    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        logging.info("Vector store criado com sucesso. Salvando em disco...")
        vectorstore.save_local(store_path)
        logging.info(f"Vector store salvo em: {store_path}")
        return vectorstore
    except Exception as e:
        logging.error(f"Erro ao criar ou salvar o vector store: {e}", exc_info=True)
        raise

def initialize_llm(model_name: str = MODEL_NAME, temperature: float = 0.0) -> ChatOpenAI:
    """
    Inicializa o modelo de linguagem (LLM).

    Args:
        model_name: Nome do modelo OpenAI a ser usado.
        temperature: Controla a aleatoriedade da resposta (0.0 = mais determinístico).

    Returns:
        O objeto ChatOpenAI inicializado.
    """
    logging.info(f"Inicializando LLM: {model_name}")
    # Temperature=0.0 para respostas mais factuais e menos criativas
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    return llm

def initialize_embeddings(embedding_model_name: str = EMBEDDING_MODEL_NAME) -> OpenAIEmbeddings:
    """
    Inicializa o modelo de embeddings.

    Args:
        embedding_model_name: Nome do modelo de embedding OpenAI.

    Returns:
        O objeto OpenAIEmbeddings inicializado.
    """
    logging.info(f"Inicializando embeddings: {embedding_model_name}")
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    # Para usar modelo open-source (ex: Sentence Transformers)
    # logging.info(f"Inicializando embeddings: {EMBEDDING_MODEL_NAME_HF}")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME_HF,
    #     model_kwargs={'device': 'cpu'} # ou 'cuda' se disponível
    # )
    return embeddings

def setup_qa_chain(llm, vectorstore) -> RetrievalQA:
    """
    Configura a cadeia RAG (RetrievalQA).

    Args:
        llm: O modelo de linguagem inicializado.
        vectorstore: O vector store FAISS inicializado.

    Returns:
        A cadeia RetrievalQA configurada.
    """
    logging.info("Configurando a cadeia de RetrievalQA...")

    # Template do Prompt - Importante para guiar o LLM
    prompt_template = """
    Você é um assistente técnico ultra-preciso. Sua ÚNICA fonte de informação é o 'Contexto' fornecido abaixo, extraído de normas técnicas. Responda à 'Pergunta' do usuário.

    Contexto:
    {context}

    Pergunta: {question}

    REGRAS ESTRITAS PARA A RESPOSTA:
    1.  Analise o contexto cuidadosamente.
    2.  Baseie sua resposta **EXCLUSIVAMENTE** nas informações presentes no 'Contexto'.
    3.  **NÃO** adicione NENHUMA informação externa, conhecimento prévio ou suposições.
    4.  Se a resposta completa e precisa para a 'Pergunta' puder ser encontrada no 'Contexto', forneça-a de forma **CLARA** e **OBJETIVA**.
    5.  Se a informação necessária para responder à 'Pergunta' **NÃO** estiver explicitamente no 'Contexto', responda **EXATAMENTE** e **SOMENTE** com a frase: "Não encontrei essa informação nas normas disponíveis." Não tente inferir, adivinhar ou dar respostas parciais se a informação completa não estiver lá.
    6.  Ao final da sua resposta (APENAS se encontrou a informação), cite o(s) documento(s) de origem usando o nome do arquivo e seção presente nos metadados do contexto. Formate a citação como: (Fonte: nome_do_arquivo_1.pdf, nome_do_arquivo_2.pdf / Seção: nome_ou_numero_da_seção)

    Resposta:
    """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Cria o Retriever a partir do Vector Store
    # search_kwargs={'k': 4} busca os 4 chunks mais relevantes
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    # Cria a cadeia RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" junta todos os chunks recuperados no prompt
        retriever=retriever,
        return_source_documents=True, # Importante para acessar os metadados da fonte
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    logging.info("Cadeia RetrievalQA configurada com sucesso.")
    return qa_chain

def ask_question(chain: RetrievalQA, query: str) -> str:
    """
    Executa uma consulta na cadeia RAG e formata a resposta.

    Args:
        chain: A cadeia RetrievalQA configurada.
        query: A pergunta do usuário em linguagem natural.

    Returns:
        A resposta formatada do assistente.
    """
    logging.info(f"Processando consulta: {query}")
    try:
        result = chain.invoke({"query": query}) # Use invoke para a versão mais recente do LangChain

        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        # Verifica se a resposta do LLM foi a específica de "não encontrado"
        if answer == "Não encontrei essa informação nas normas disponíveis.":
             logging.info("Resposta gerada: Informação não encontrada.")
             return answer

        # Se encontrou a resposta, formata incluindo as fontes
        if source_docs:
            # Extrai nomes únicos dos arquivos de origem dos metadados
            sources = set()
            for doc in source_docs:
                # O nome do arquivo geralmente está em 'source' nos metadados do PyPDFLoader
                source_name = doc.metadata.get('source', 'desconhecida')
                # Extrai apenas o nome do arquivo do caminho completo
                sources.add(os.path.basename(source_name))

            if sources:
                source_citation = ", ".join(sorted(list(sources)))
                # Adiciona a citação ao final da resposta, caso o LLM não tenha adicionado
                # (embora o prompt peça, é bom garantir)
                if f"(Fonte: {source_citation})" not in answer:
                     final_answer = f"{answer}\n(Fonte: {source_citation})"
                else:
                     # Se o LLM já adicionou, ajusta para garantir o formato
                     # Esta parte pode precisar de ajuste dependendo de como o LLM formata
                     import re
                     answer = re.sub(r"\s*\(Fonte: [^\)]+\)\s*$", "", answer) # Remove citação existente no final
                     final_answer = f"{answer.strip()}\n(Fonte: {source_citation})"

                logging.info(f"Resposta gerada: {final_answer}")
                return final_answer
            else:
                 # Caso encontre resposta mas não consiga extrair fonte (improvável com setup atual)
                 logging.warning("Resposta gerada, mas fontes não encontradas nos metadados.")
                 return f"{answer}\n(Fonte: Origem não identificada nos metadados)"
        else:
            # Se o LLM deu uma resposta mas não retornou source_docs (improvável com return_source_documents=True)
            logging.warning("Resposta gerada pelo LLM, mas sem documentos fonte retornados.")
            return answer # Retorna a resposta do LLM como está

    except Exception as e:
        logging.error(f"Erro ao processar a consulta: {e}", exc_info=True)
        return "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."

# --- Função Principal ---
def main():
    """
    Função principal que orquestra a inicialização e execução do assistente.
    """
    logging.info("Iniciando o Assistente de Normas Técnicas...")

    # 1. Carregar documentos
    documents = load_documents(DOCS_PATH)
    if not documents:
        logging.error(f"Nenhum documento encontrado ou carregado de '{DOCS_PATH}'. Encerrando.")
        print(f"Erro: Verifique se a pasta '{DOCS_PATH}' existe e contém arquivos PDF válidos.")
        return

    # 2. Dividir documentos em chunks
    split_docs = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not split_docs and os.path.exists(VECTORSTORE_PATH):
        logging.warning("Nenhum documento novo para processar, mas um vector store existente foi encontrado e será carregado.")
    elif not split_docs:
        logging.error("Falha ao dividir documentos e nenhum vector store existente encontrado. Encerrando.")
        print("Erro: Falha ao processar os documentos. Verifique os arquivos PDF.")
        return

    # 3. Inicializar embeddings
    try:
        embeddings = initialize_embeddings()
    except Exception as e:
        logging.error(f"Falha ao inicializar embeddings: {e}. Verifique a configuração da API ou do modelo local.", exc_info=True)
        print("Erro: Falha ao conectar com o serviço de embeddings. Verifique sua chave de API ou modelo local.")
        return

    # 4. Criar ou Carregar Vector Store
    try:
        # Passamos os split_docs apenas se precisarmos criar um novo índice
        # Se o índice já existir, split_docs não são necessários para carregá-lo
        vectorstore = create_or_load_vectorstore(split_docs if not os.path.exists(VECTORSTORE_PATH) else [], embeddings, VECTORSTORE_PATH)
    except ValueError as ve: # Captura erro específico de não ter docs para criar novo VS
         logging.error(f"Erro ao configurar o vector store: {ve}")
         print(f"Erro: {ve}")
         return
    except Exception as e:
        logging.error(f"Erro ao configurar o vector store: {e}", exc_info=True)
        print("Erro: Falha ao criar ou carregar o banco de dados vetorial.")
        return


    # 5. Inicializar LLM
    try:
        llm = initialize_llm()
    except Exception as e:
        logging.error(f"Falha ao inicializar LLM: {e}. Verifique a chave da API OpenAI.", exc_info=True)
        print("Erro: Falha ao conectar com a API da OpenAI. Verifique sua chave.")
        return

    # 6. Configurar a cadeia RAG
    qa_chain = setup_qa_chain(llm, vectorstore)

    # 7. Loop Interativo para Consultas
    print("\n--- Assistente de Normas Técnicas ---")
    print("Digite sua pergunta sobre as normas carregadas. Digite 'sair' para terminar.")

    while True:
        query = input("\nSua pergunta: ")
        if query.lower().strip() == 'sair':
            print("Encerrando o assistente. Até logo!")
            break
        if not query.strip():
            print("Por favor, digite uma pergunta.")
            continue

        response = ask_question(qa_chain, query)
        print(f"\nAssistente: {response}")

if __name__ == "__main__":
    main()