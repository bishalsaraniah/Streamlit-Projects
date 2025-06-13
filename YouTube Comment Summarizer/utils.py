import streamlit as st
import openai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

gemini_api_key = st.secrets['GEMINI_API_KEY']
openai.api_key = st.secrets['OPENAI_API_KEY']
load_dotenv()
'''
def get_summary(text):
    """Generates yes a summary of the given text using OpenAI's GPT-4."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the latest model available
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this: {text}"}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "An error occurred while generating the summary."'''

def get_summary(text):
    #Tokenization
    text_splitter = TokenTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10
    )
    chunks = text_splitter.create_documents([text])

    #Summarization
    llm = ChatGoogleGenerativeAI(
        # model="gemini-pro",
        # model="gemini-2.5-flash",
        # model="gemini-1.5-pro",
        # model="gemini-1.5-flash-8b-001",
        
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key
    )
                                                                
    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce"
    )

    #Invoke Chain
    response=chain.run(chunks)

    return response