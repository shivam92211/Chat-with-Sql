import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY is missing")
else:
    # Configure the API with the key
    genai.configure(api_key=api_key)

# Initialize the Google Generative AI model
model = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    temperature=0.3
)

class ChatWithSql:
    """
    ChatWithSql class is used for chat and query user question with the SQL database.
    """
    def __init__(self, db_user, db_password, db_host, db_name):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_name = db_name
        self.llm = model
        
    def message(self, query):
        db = SQLDatabase.from_uri(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=True
        )
        try:
            response = agent_executor.run(query)
        except Exception as e:
            return f"Error: {str(e)}"
        
        return response

# Streamlit app layout
st.title("AI based SQL system")

# User input fields
db_user = st.text_input("Database User")
db_password = st.text_input("Database Password", type="password")
db_host = st.text_input("Database Host")
db_name = st.text_input("Database Name")
query = st.text_area("Enter you questio related to Database")

# Button to send message
if st.button("Send Query"):
    if db_user and db_password and db_host and db_name and query:
        chat_obj = ChatWithSql(
            db_user=db_user,
            db_password=db_password,
            db_host=db_host,
            db_name=db_name
        )
        response = chat_obj.message(query)
        st.success(f"Response: {response}")
    else:
        st.warning("Please fill in all fields.")
