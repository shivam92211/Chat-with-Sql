from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
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

model = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    temperature=0.3)

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class QueryRequest(BaseModel):
    db_user: str
    db_password: str
    db_host: str
    db_name: str
    query: str

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
            raise HTTPException(status_code=400, detail=str(e))

        return response

@app.post('/send-message')
async def send_message(request: QueryRequest):
    chat_obj = ChatWithSql(
        db_user=request.db_user,
        db_password=request.db_password,
        db_host=request.db_host,
        db_name=request.db_name
    )
    response = chat_obj.message(request.query)
    return {"message": response}
