import hashlib
import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(".env 파일 또는 Streamlit Secrets에 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

SYSTEM_PROMPT = """너는 업로드된 문서를 근거로 답변하는 RAG 챗봇이다.
반드시 제공된 context 안의 내용만 근거로 답변하라.
context에 답이 없으면 '업로드된 문서에서 찾을 수 없습니다'라고 답하라.
내용을 지어내거나 문서에 없는 정보를 추측하지 말라."""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "context:\n{context}\n\n질문: {question}"),
    ]
)


@st.cache_resource(show_spinner=False)
def build_rag_components(file_bytes, file_hash, api_key):
    temp_pdf_path = None
    persist_directory = os.path.join(
        tempfile.gettempdir(), f"streamlit_rag_{file_hash}"
    )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_pdf_path = temp_file.name

        documents = PyPDFLoader(temp_pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )
        chain = chat_prompt | llm | StrOutputParser()
        return retriever, chain
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def page_number(document):
    page = document.metadata.get("page")
    return page + 1 if isinstance(page, int) else "?"


def format_docs(documents):
    parts = []
    for index, document in enumerate(documents, start=1):
        parts.append(
            f"[청크 {index} | PDF {page_number(document)}페이지]\n"
            f"{document.page_content}"
        )
    return "\n\n".join(parts)


def document_sources(documents):
    return [
        {
            "page": page_number(document),
            "content": document.page_content,
        }
        for document in documents
    ]


def render_sources(sources):
    with st.expander("참고한 본문 보기"):
        for source in sources:
            st.markdown(
                f"**PDF {source['page']}페이지**\n\n{source['content']}"
            )


st.set_page_config(page_title="PDF 문서 RAG 챗봇", page_icon="📄")
st.title("PDF 문서 RAG 챗봇")
uploaded_file = st.file_uploader("질문할 PDF를 업로드하세요.", type=["pdf"])

if uploaded_file is None:
    st.info("PDF를 업로드하면 문서 검색과 채팅을 시작할 수 있습니다.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.messages = []
    st.session_state.pop("retriever", None)
    st.session_state.pop("chain", None)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("PDF를 처리하고 있습니다..."):
    retriever, chain = build_rag_components(file_bytes, file_hash, OPENAI_API_KEY)

st.session_state.retriever = retriever
st.session_state.chain = chain

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("업로드된 문서에 대해 질문하세요.")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            documents = retriever.invoke(question)
            context = format_docs(documents)

        with st.spinner("답변 생성 중..."):
            answer = chain.invoke({"context": context, "question": question})

        st.markdown(answer)
        sources = document_sources(documents)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
