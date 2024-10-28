import getpass
import os


from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver


from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from langchain_core.messages import SystemMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit

if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
os.environ["AZURE_OPENAI_ENDPOINT"]="https://hassanllms.openai.azure.com/"


LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="pr-aching-chess-83"


from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # or your deployment
    api_version="2024-02-15-preview",  # or your api version
    temperature=0,
    # other params...
    )


engine = create_engine("sqlite:///CompanyDB.db")

db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())



toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()


SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input , create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.You are an intelligent interview bot conducting a maturity assessment. You have access to a SQLite database with a 'CompanyDBTable' table (columns: Department, Category, Question).


IMPORTANT: You must ONLY ask questions that were retrieved from the database. Do not generate new questions.

IMPORTANT INSTRUCTIONS:
1. When you first interact with a user:
   - Extract their name, email, and department from their message
   - Use this SQL query to get all questions for their department:
     SELECT Category, Question FROM MaturityCheck WHERE Department = ? ORDER BY Category;
   - Store these questions in your context for the conversation

2. After getting the questions:
   - Ask questions one at a time
   - Wait for the user's response before asking the next question
   - Group questions by category and introduce each category
   - Keep track of which questions you've asked
   - Don't repeat questions that have been answered

3. For each response:
   - Briefly acknowledge the answer
   - Move to the next question
   - When finishing a category, introduce the next one"""

system_message = SystemMessage(content=SQL_PREFIX)






def get_agent():
    # Your existing agent setup code here
    memory = MemorySaver()
    
    agent_executor = create_react_agent(llm, tools, messages_modifier=system_message,
                                      checkpointer=memory)
    
    config = {"configurable": {"thread_id": "abc123"}}
    
    return agent_executor, config


