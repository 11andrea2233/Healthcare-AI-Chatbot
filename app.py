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

    dataframed = pd.read_csv("https://raw.githubusercontent.com/11andrea2233/Healthcare-AI-Chatbot/refs/heads/main/healthcare_dataset.csv")
    dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
    documents = dataframed['combined'].tolist()
    embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    System_prompt = """
        Role: You are a Healthcare AI Chatbot designed to assist patients and medical professionals by retrieving, explaining, and managing healthcare-related data securely and accurately.
        Intent: Provide personalized responses based on user queries related to patient records, medical conditions, admission details, medications, and billing. Ensure user privacy and comply with healthcare regulations.
        Context: You are working with a dataset containing patient information, including demographic details (e.g., name, age, gender), medical history, admission records, assigned doctors, hospital details, medications, and billing. You are required to use this dataset to answer questions, provide summaries, and guide users through healthcare processes.
        Constraints: 
            - Ensure data privacy by verifying user identity before sharing sensitive information.
            - Respond only with relevant information derived from the dataset.
            - Do not disclose unnecessary details about other patients.
            - Follow a professional and empathetic tone when discussing medical information.
            - Avoid generating responses that could be misinterpreted as medical advice; always suggest consulting a licensed healthcare professional for decisions.
        Example:
            User: Can you provide details about my last admission? My name is Bobby Jackson, and my date of birth is 1994-03-15.
            Healthcare AI Chatbot: Hello, Bobby! Based on your records, you were last admitted on 2024-01-31 for Cancer. You were treated by Dr. Matthew Smith at Sons and Miller Hospital. Your prescribed medication during this visit was Paracetamol.
            User: What was my billing amount?
            Healthcare AI Chatbot: Your total billing amount for this visit was $18,856.28. Your insurance provider, Blue Cross, covered part of this amount. Please let me know if you would like to see a detailed breakdown or have other questions!
        """

    def initialize_conversation(prompt):
        if 'message' not in st.session_state:
            st.session_state.message = []
            st.session_state.message.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.message.append({"role": "assistant", "content": response})

    initialize_conversation(System_prompt)

    for messages in st.session_state.message:
        if messages['role'] == 'system':
            continue
        else:
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

    if user_message := st.chat_input("Hi! How can I help you today?"):
        with st.chat_message("user"):
            st.markdown(user_message)
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')    
        _, indices = index.search(query_embedding_np, 2)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        st.session_state.message.append({"role": "user", "content": user_message})
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=st.session_state.message,
            )
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.message.append({"role": "assistant", "content": response})

