import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os
import speech_recognition as sr

# Streamlit theme (Optional)
st.set_page_config(page_title="LLM Chat App", page_icon=":robot:")

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    
    ''')
    add_vertical_space(5)
    st.write('Made by Rohan')

load_dotenv()


def record_audio():
  recognizer = sr.Recognizer()
  with sr.Microphone() as source:
      print("Speak Anything...")
      audio = recognizer.listen(source)
  try:
      text = recognizer.recognize_google(audio)
      print("You said: " + text)
      return text
  except sr.UnknownValueError:
      print("Could not understand audio")
      return None
  except sr.RequestError as e:
      print("Could not request results from Google Speech Recognition service; {0}".format(e))
      return None


def main():
    # User Interface Elements
    st.header("Chat with your PDF")

    # Upload area with a custom message
    pdf = st.file_uploader("Upload your PDF and ask questions:", type='pdf')

    # Display loading indicator while processing PDF
    with st.spinner('Processing your PDF...'):
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

    # Voice Search Button
    voice_search_button = st.button("Search by Voice")

    # Initialize voice_query outside the if block
    voice_query = None  

    if voice_search_button:
        voice_query = record_audio()
        if voice_query:
            st.write("You searched for:", voice_query)
            # Use voice_query for further processing

    # Text Input with Autocomplete (Optional)
    # Commenting out for now as we're using voice search
    # query = st.text_input("Ask me anything about the PDF:")

    # Use voice_query instead of text input for further processing
    if voice_query:
        store_name = pdf.name[:-4]

        # Embeddings
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        docs = VectorStore.similarity_search(query=voice_query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=voice_query)
            print(cb)
        st.write(response)


if __name__ == '__main__':
    main()
