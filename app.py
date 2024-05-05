
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

import streamlit as st

from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()

import re

##! Converting PDF to text ## 
# def read_pdf(pdf):
#     pdf_reader = PdfReader(pdf)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

##! Saving vectorestore locally ##
# def save_vectorstore(title, vector_store):
#     title = re.sub('\s+', '-', title)
#     title = re.sub('[^a-zA-Z0-9\-]+', '', title)
#     vector_store.save_local(f"./vectordb/{title[:35]}")
#     print(f"{title} stored!")
#     return True

##* Load vectorstore ##
def load_vectorstore(name, embeddings):
    vector_store = FAISS.load_local(f"./vectordb/{name}/", embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_store
    
def main():
    st.header('Chat with PDF 💬')

    # pdf = st.file_uploader("Upload PDF", type='pdf')


    embeddings = OpenAIEmbeddings()
    vector_store = ''

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    # st.button('Load', on_click=click_button)
    with st.container(border=True):
        st.markdown('''
             *Disclaimer: Section for uploading the PDF file has been removed as the API calls for OpenAI are not free. I've included few pdfs for Q&A. You can access the source code and enable the section for uploading PDFs.*
                ''')
    
    ##! Converting text to word Embeddings ##
    # if st.session_state.clicked:
    #     if pdf is not None:
    #         text = read_pdf(pdf)
    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=1000,
    #             chunk_overlap=200,
    #             length_function=len
    #         )
    #         chunks = text_splitter.split_text(text=text)
    #         vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    #         if save_btn:
    #             save_vectorstore(pdf.name[:-4], vector_store)
            


    option = st.selectbox(label="Select the PDF: ", options=['Budget Speech 2024', 'The 100 Page Machine Learning Book'], index=None)

    query = st.text_input("Ask questions from your PDF file:")

    if query and not option:
        st.warning("Please Select a PDF")

    if option:
        title = re.sub('\s+', '-', option)
        title = re.sub('[^a-zA-Z0-9\-]+', '', title)
        vector_store = load_vectorstore(title, embeddings)

        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.invoke({"input_documents": docs, "question":query})
                print(cb)
                st.write(response["output_text"])


if __name__ == '__main__':
    main()