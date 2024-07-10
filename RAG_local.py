import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #AutoModel, 
from sentence_transformers import SentenceTransformer
import logging
import shutil
from langchain.text_splitter import CharacterTextSplitter


logging.basicConfig(level=logging.INFO)

# 기존 데이터베이스 삭제 함수
def clear_database(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)


# Local model path(로컬 모델 경로 설정)
local_embedding_model_path = "./models/klue-roberta-base"
local_llm_model_path = "./models/skt/kogpt2-base-v2"  # 한국어 sLLM 모델

#from langchain.document_loaders import TextLoader, PyPDFLoader

#local_model_path = "./models/klue-roberta-base"
embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)

# # 벡터 데이터베이스 초기화
# vectorstore = Chroma(
#     collection_name="multilingual_collection",
#     embedding_function=embeddings,
#     client=chromadb.Client(Settings(allow_reset=True))
# )


# Excel 파일을 배치로 로드하는 함수
def load_excel_in_batches(file_path, batch_size=1000):
    # Excel 파일 전체를 읽습니다
    df = pd.read_excel(file_path)
    total_rows = len(df)
    
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        chunk = df.iloc[start:end]
        
        documents = []
        for _, row in chunk.iterrows():
            content = " ".join(row.astype(str).values)
            doc = Document(page_content=content, metadata={"source": file_path})
            documents.append(doc)
        
        yield documents
        
# 문서 분할 함수
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 문서 로딩 함수
def load_documents(directory, vectorstore):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents = loader.load()
            documents = split_documents(documents)
            vectorstore.add_documents(documents)
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            documents = split_documents(documents)
            vectorstore.add_documents(documents)
        elif filename.endswith('.xlsx'):
            for batch in load_excel_in_batches(file_path):
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                vectorstore.add_texts(texts, metadatas=metadatas)
    return vectorstore


# main
if __name__ == "__main__":
    logging.info(f"Using embedding model from: {local_embedding_model_path}")
    logging.info(f"Using LLM model from: {local_llm_model_path}")
    
    # persist_directory = "/path/to/save/vectorstore"
    # # 기존 데이터베이스 삭제
    # clear_database(persist_directory)
    
    input_directory = r""   #"/path/to/your/documents"  # 문서가 있는 폴더 경로
    persist_directory = r""  #"/path/to/save/vectorstore"  # 벡터 데이터베이스를 저장할 경로

    # # 로컬 임베딩 모델 초기화
    # embedding_model = SentenceTransformer(local_embedding_model_path)
    # embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
    
    
    # Initialize Embedding model(임베딩 모델 초기화)
    local_embedding_model_path = "./models/klue-roberta-base"
    if not os.path.exists(local_embedding_model_path):
        embedding_model = SentenceTransformer('klue/roberta-base')
        embedding_model.save(local_embedding_model_path)
    else:
        embedding_model = SentenceTransformer(local_embedding_model_path)
        
    # 임베딩 차원 확인
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

    # HuggingFaceEmbeddings 초기화
    embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
    
    # Initialize Chroma vectorDB (Chroma 벡터 데이터베이스 초기화 (차원 명시))
    vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine", "dimension": embedding_dim}
)

    # load documents and Create vectorDB(문서 로딩 및 벡터 데이터베이스 생성)
    vectorstore = load_documents(input_directory, vectorstore)
    vectorstore.persist()

# Initialize LLM(LLM 초기화)
    tokenizer = AutoTokenizer.from_pretrained(local_llm_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_llm_model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Difine Prompt template(프롬프트 템플릿 정의)
    template = """
    Context: {context}
    Question: {question}
    Answer the question based on the context provided.
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create LLM chain(LLM 체인 생성)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    def get_response(query):
        docs = vectorstore.similarity_search(query)
        
        # Restrict Context length (컨텍스트 길이 제한)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        context = text_splitter.split_text(" ".join([doc.page_content for doc in docs]))
        context = context[0] if context else ""  # 첫 번째 청크만 사용
        
        response = llm_chain.invoke({"context": context, "question": query})
        return response['text']

    # Chatbot Interface(챗봇 인터페이스)
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = get_response(user_input)
        print("Bot:", response)
