import base64
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain.chains import AnalyzeDocumentChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.loading import load_llm
from langchain.memory import ChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from llama_index import download_loader

# from langchain.chat_models import ChatOpenAI
# from llama_index import (
#     QuestionAnswerPrompt,
#     RefinePrompt,
#     GPTSimpleVectorIndex,
#     LLMPredictor,
#     download_loader,
# )


def show_pdf(file_path: str):
    """Show the PDF in Streamlit
    That returns as html component

    Parameters
    ----------
    file_path : [str]
        Uploaded PDF file path
    """

    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf">'  # noqa: E501
    return pdf_display


st.title("PDF Summary")

st.write("Enter your OpenAI API key")
openai_api_key = st.text_input("OpenAI API Key", value="", type="password")

response_language = st.selectbox(
    "Language of response", ("English", "Japanese")
)

if "qa" not in st.session_state:
    st.session_state["qa"] = None
if "history" not in st.session_state:
    st.session_state["history"] = None

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.write("API keys have been set.")

    uploaded_file = st.file_uploader("Upload pdf", type="pdf")

    if uploaded_file and st.button("Preprocess data"):
        with st.spinner("Preprocessing data..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                fp = Path(tmp_file.name)
                fp.write_bytes(uploaded_file.getvalue())
                st.write(show_pdf(tmp_file.name))

                # PDFReader = download_loader("PDFReader")
                # loader = PDFReader()
                # documents = loader.load_data(file=fp)
                llm = load_llm("llm.json")
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=0, separator="\n"
                )
                texts = text_splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings()
                embeddings_filter = EmbeddingsFilter(
                    embeddings=embeddings, similarity_threshold=0.76
                )
                # if not Path("./faiss_index").exists():
                #     db = FAISS.from_documents(texts, embeddings)
                #     db.save_local("faiss_index")
                with tempfile.NamedTemporaryFile(
                    delete=False
                ) as tmp_faiss_index:
                    db_fp = Path(tmp_faiss_index.name)
                    db = FAISS.from_documents(texts, embeddings)
                    db.save_local(str(db_fp))

                docsearch = FAISS.load_local("faiss_index", embeddings)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=embeddings_filter,
                    base_retriever=docsearch.as_retriever(),
                )

                prompt_template = """
Use the following pieces of context to answer the question at the end. This context is derived from a paper from a user. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in $response_language$:
"""
                prompt_template.replace(
                    "$response_language$", response_language
                )
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"],
                )
                chain_type_kwargs = {"prompt": PROMPT}

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=compression_retriever,
                    chain_type_kwargs=chain_type_kwargs,
                )
                # history = ChatMessageHistory()

                st.success("Data loaded successfully!")

                st.session_state.qa = qa
                # st.session_state.history = history

# if response_language:
#     prompt_template = """
# Use the following pieces of context to answer the question at the end. This context is derived from a paper from a user. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Answer in $response_language$:
# """
#     prompt_template.replace("$response_language$", response_language)
#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#     chain_type_kwargs = {"prompt": PROMPT}

if st.session_state.qa:
    query_str = st.text_input("Enter your question:", value="")

    if query_str != "":
        response = st.session_state.qa.run(query=query_str)
        st.write("Response:")
        st.markdown(
            f"<h3 style='font-size: 18px;'>{response}</h3>",
            unsafe_allow_html=True,
        )

        # st.write("Source:")
        # st.markdown(
        #     f"<h3 style='font-size: 18px;'>{output.source_nodes[0].source_text[:100]}...</h3>",
        #     unsafe_allow_html=True,
        # )

        # if st.session_state.history:
        #     st.session_state.history.add_user_message(query_str)
        #     st.session_state.history.add_ai_message(response)
