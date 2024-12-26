import streamlit as st
from streamlit_option_menu import option_menu
import openai
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Your 24/7 Healthcare Receptionist", page_icon="👨‍⚕️👩‍⚕️", layout="wide")

with st.sidebar:
    openai.api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning("Please enter your OpenAI Key to proceed.", icon="🔔")
    else:
        st.success("How can I help you today?", icon="🩺")
        
    options = option_menu(
        "Dashboard",
        ["Home", "About Me", "Healthcare AI Chatbot"],
        icons=['house', "🩺",  'file-text'],
        menu_icon="list",
        default_index=0
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

# Options: Home
if options == "Home":
    st.title ("Welcome to HealthCare AI Chatbot")
    st.write("Hello there! I am your new AI receptionist, designed to make life easier for both healthcare providers and patients. Think of me as a virtual assistant that can help answer patient questions, schedule appointments, provide medication reminders, handle basic billing inquiries, and share information about the clinic. I am here to make sure patients get the answer they need quickly and securely, while freeing up staff to focus on more important tasks.")
    st.write("What makes me special? Well, I use advanced technology to look up the right information and respond accurately, like an efficient, friendly receptionist. I am always available and ready to assist, whether its early in the morning or late at night.")
        
    st.write("What can I do?")
    st.write("1. Scheduling Appointments. You can book, reschedule or cancel appointments without having to wait on hold or visit the clinic in person.")
    st.write("2. Medication Reminders. I can remind patients of their current medications and help them request refills, saving everyone time.")
    st.write("3. Billing Questions. For simple billing questions, I can look up amounts and payment dates, and if things get too complicated, it will direct you to the billing team.")
    st.write("4. General Clinic Information. Whether its clinic hours, location or accepted insurance plans, i have all the details to keep patients informed.")
        
    st.write("Purpose")
    st.write("My job is simple: to take on the routine questions so your healthcare team can focus on what matters most—caring for patients! Here is how I help:")
    st.write("Save Time: By handling common questions, I give staff more time to focus on patient care.")
    st.write("Answer Quickly: Patients dont have to wait long for answers about appointments, medications, and billing—I can respond almost instantly!")
    st.write("Protect Privacy: I make sure all patient information is kept safe and secure, following healthcare privacy rules.")
    
elif options == "About Me":
    st.header("About Me")
    st.markdown("""
        Hi! I'm Andrea Arana! I am a business intelligence specialist and data analyst with a strong foundation in deriving insights from data to drive strategic decisions. Currently, I am expanding my skill set by learning to create products in Artificial Intelligence and working towards becoming an AI Engineer. My goal is to leverage my expertise in BI and data analysis with advanced AI techniques to create innovative solutions that enhance business intelligence and decision-making capabilities. 
        
        This projects is one of the projects I am building to try and apply the learning I have acquired from the AI First Bootcamp of AI Republic.
        
        Any feedback would be greatly appreciated! ❤           
                    """)
    st.text("Connect with me on LinkedIn 😊 [Andrea Arana](https://www.linkedin.com/in/andrea-a-732769168/)")

elif options == "Healthcare AI Chatbot":
    st.header("Healthcare AI Chatbot")
    st.write("This is a chatbot that can answer questions about the clinic, schedule appointments, provide medication reminders, handle basic billing inquiries, and share information about the clinic.")
    st.write("You can ask me about the clinic, schedule appointments, request medication refills, or check your billing information.")
    st.write("I am always here to help you!")

    dataframed = pd.read_csv("data/healthcare_data.csv")
    dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
    documents = dataframed['combined'].tolist()
    embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)