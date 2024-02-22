import streamlit as st
from PyPDF2 import PdfReader
from modules import *
from PIL import Image
import io
import pathlib
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import faiss

st.set_page_config(layout = 'wide')
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)

if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'tutor_button' not in st.session_state:
    st.session_state.tutor_button = False

st.header("AI Tutor - Coming March")
choice = st.radio("Select the feature you want to try", ["AI Art Tutor 🎓", "AI Art Vision 🎨"], index=0)

col1, col2 = st.columns(2)
if choice == "AI Art Vision 🎨":
    model = genai.GenerativeModel('gemini-pro-vision')
    img = st.file_uploader("Upload the image for analysis", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if img: 
        image = Image.open(io.BytesIO(img.read())) 
        st.image(image, caption='Uploaded Image.', use_column_width=True, width = 100)
        response = model.generate_content(['''You are a professional artist and critic, please provide a detailed analysis of the artwork. Keep your response no less than 500 tokens, use markdown for nice formatting''', image], stream=True)
    if st.button("Tell me more about this art 🎨") and image:
        response.resolve()
        st.write(response.text)
    

elif choice == "AI Art Tutor 🎓": 
    pdf = st.file_uploader("Upload PDF file and click the button", type="pdf", accept_multiple_files=False)
    if st.button("Process Documents and Ask AI Tutor 🎓"):
        st.session_state.tutor_button = True
    if st.session_state.tutor_button and pdf:
        retrieval_augmented_generation(pdf)
        query = st.text_input("Please type your questions on the textbook:")
        if query:
            #st.success("You have entered a query")
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_document",
                title="Embedding of single string")
            st.toast("Query converted to Embeddings")
            _, indices = st.session_state.faiss_index.search(np.array([result['embedding']]), 3)
            # st.write(indices) 
            # st.write("The AI Tutor 🎓 says:")
            # for idx in indices[0]:
            #   st.write(st.session_state.text_chunks[idx])
            model = genai.GenerativeModel('gemini-pro')
            material = []
            for idx in indices[0]:
                material.append(st.session_state.text_chunks[idx])
            prompt=  f"You are a professional art educator, the student has asked {query}, and you need to provide a detailed response to the query. I've provided some materials, they might be irrelevant at all, you need to see if they help answer the question better." 
            response = model.generate_content([prompt, ' '.join(material)], stream=True)
            response.resolve()
            st.write(response.text)


with st.expander("Important Information",expanded=False):
    st.header("AI Tutor - Coming March")

    st.info("Our technical team is deeply committed to developing this new feature. Due to its complexity and our ongoing efforts to gather more data, we anticipate fully launching this feature by March.")

    st.markdown(
        '''
        **AI Art Tutor 🎓** will feature the below features:
        1. **AI Art Tutor 🎓**:
        - **Reliable Knowledge Bot**: Utilizing Gemini's AI Language Services, our chatbot is designed to deliver trustworthy responses to users' inquiries. At the core of this service is the Retrieval-Augmented Generation (RAG) technology, 
        which functions akin to an exceptionally knowledgeable digital librarian. This system meticulously searches through an extensive and credible database of textbooks and scholarly articles, ensuring that the information provided is 
        detailed and accurate.

        - **Voice Assitant**: Implementing Google's Speech services for high-accuracy speech-to-text, text-to-speech, and real-time translation features to cater to a wider, more diverse audience.
        We also plan to create AI avatars of renowned artists crafted from historical records of their personalities and personal stories. This interactive Q&A feature with the artists will enhance the educational enjoyment for users. 
        Moreover, our multilingual translations will ensure inclusivity, offering a seamless experience for a global audience.

        - **Artwork Advice & Inspiration**: Utilizing Google's AI capabilities, including Gemini's vision and Imagen's image generation, to offer personalized advice and creative inspirations based on user-submitted artwork and requirements.
        
    '''
    )
     


