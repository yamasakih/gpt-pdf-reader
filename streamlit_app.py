import base64
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

if "index" not in st.session_state:
    st.session_state["index"] = None

if openai_api_key:
    st.write("API keys have been set.")

    uploaded_file = st.file_uploader("Upload pdf", type="pdf")

    if uploaded_file and st.button("Compress data"):
        with st.spinner("Loading data..."):
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
                print(f"{texts=}")

                embeddings = OpenAIEmbeddings()
                embeddings_filter = EmbeddingsFilter(
                    embeddings=embeddings, similarity_threshold=0.76
                )
                if not Path("./faiss_index").exists():
                    db = FAISS.from_documents(texts, embeddings)
                    db.save_local("faiss_index")
                docsearch = FAISS.load_local("faiss_index", embeddings)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=embeddings_filter,
                    base_retriever=docsearch.as_retriever(),
                )
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=compression_retriever,
                )
                history = ChatMessageHistory()

                st.success("Data loaded successfully!")

                st.session_state.qa = qa
                st.session_state.history = history
                # llm_predictor = LLMPredictor(
                #     llm=ChatOpenAI(
                #         temperature=0,
                #         model_name="gpt-3.5-turbo",
                #         openai_api_key=openai_api_key,
                #     )
                # )
                # st.session_state.index = GPTSimpleVectorIndex(
                #     documents, llm_predictor=llm_predictor
                # )

# if response_language:
#     QA_PROMPT_TMPL = (
#         "We have provided context information below. \n"
#         "---------------------\n"
#         "{context_str}"
#         "\n---------------------\n"
#         f"Given this information, please answer the question in {response_language}\n"
#         "question: {query_str}\n"
#     )
#     REFINE_PROMPT_TMPL = (
#         "The original question is as follows: {query_str}\n"
#         "We have provided an existing answer: {existing_answer}\n"
#         "We have the opportunity to refine the existing answer "
#         "(only if needed) with some more context below.\n"
#         "------------\n"
#         "{context_msg}\n"
#         "------------\n"
#         f"Given the new context, refine the original answer to better in {response_language}"
#         "answer the question. "
#         "If the context isn't useful, return the original answer."
#     )
#     QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
#     REFINE_PROMPT = RefinePrompt(REFINE_PROMPT_TMPL)

if st.session_state.qa:
    query_str = st.text_input("Enter your question:", value="")

    response = qa.run(query=query_str)
    # response = compression_retriever.get_relevant_documents(query_str)
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
    history.add_user_message(query_str)
    history.add_ai_message(response)

    # if query_str:
    #     output = st.session_state.index.query(
    #         query_str,
    #         text_qa_template=QA_PROMPT,
    #         refine_template=REFINE_PROMPT,
    #     )
    #     st.write("Response:")
    #     st.markdown(
    #         f"<h3 style='font-size: 18px;'>{output}</h3>",
    #         unsafe_allow_html=True,
    #     )

    #     st.write("Source:")
    #     st.markdown(
    #         f"<h3 style='font-size: 18px;'>{output.source_nodes[0].source_text[:100]}...</h3>",
    #         unsafe_allow_html=True,
    #     )
