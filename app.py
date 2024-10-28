import streamlit as st
import requests
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
import time
from agentexecutor import get_agent




# def initialize_session_state():
#     if 'user_id' not in st.session_state:
#         st.session_state.user_id = None
#     if 'conversation_active' not in st.session_state:
#         st.session_state.conversation_active = False
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

# def process_chat(agent_executor, query: str, is_first_message: bool = False):
#     if is_first_message:
#         # Create user in database only for the first message
#         user_response = requests.post(
#             "http://localhost:8000/users/",
#             json={"message": query}
#         ).json()
#         print(user_response.get("user_id"))
#         st.session_state.user_id = user_response["user_id"]
#         st.session_state.conversation_active = True
    
#     messages = []
#     for s in agent_executor.stream(
        
#     ):
        
#         if 'agent' in s and 'messages' in s['agent']:
#             message = s['agent']['messages'][0]
            
#             # If this is a question from the bot (not intermediate processing)
#             if message.content and not message.additional_kwargs.get('tool_calls'):
#                 # Save bot's message
#                 st.session_state.messages.append({"role": "assistant", "content": message.content})
                
#                 # Save the response to database
#                 if not is_first_message:  # Only save responses after the first message
#                     response = requests.post(
#                         "http://localhost:8000/responses/",
#                         json={
#                             "user_id": st.session_state.user_id,
#                             "bot_message": message.content,
#                             "user_answer": query  # This is the user's previous answer
#                         }
#                     )
#                 return message.content
#     return None

# def main():
#     st.title("Technical Assessment Chatbot")
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Your existing Azure OpenAI and agent setup code should go here
#     # agent_executor = ... (your existing setup)
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Your message"):
#         # Display user message
#         with st.chat_message("user"):
#             st.write(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Process the message
#         is_first_message = not st.session_state.conversation_active
        
#         with st.chat_message("assistant"):
#             response = process_chat(agent_executor, prompt, is_first_message)
#             if response:
#                 st.write(response)

# if __name__ == "__main__":
#     main()




import streamlit as st
import requests
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
import time
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langgraph.checkpoint.memory import MemorySaver


from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from langchain_core.messages import SystemMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import getpass
import os


# if "AZURE_OPENAI_API_KEY" not in os.environ:
#     os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
#         "Enter your AzureOpenAI API key: "
#     )
# os.environ["AZURE_OPENAI_ENDPOINT"]="https://hassanllms.openai.azure.com/"


# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY="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"
# LANGCHAIN_PROJECT="pr-aching-chess-83"


# from langchain_openai import AzureChatOpenAI

# SQL_PREFIX = """You are an agent designed to interact with a SQL database.
# Given an input , create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.You are an intelligent interview bot conducting a maturity assessment. You have access to a SQLite database with a 'CompanyDBTable' table (columns: Department, Category, Question).


# IMPORTANT: You must ONLY ask questions that were retrieved from the database. Do not generate new questions.

# IMPORTANT INSTRUCTIONS:
# 1. When you first interact with a user:
#    - Extract their name, email, and department from their message
#    - Use this SQL query to get all questions for their department:
#      SELECT Category, Question FROM MaturityCheck WHERE Department = ? ORDER BY Category;
#    - Store these questions in your context for the conversation

# 2. After getting the questions:
#    - Ask questions one at a time
#    - Wait for the user's response before asking the next question
#    - Group questions by category and introduce each category
#    - Keep track of which questions you've asked
#    - Don't repeat questions that have been answered

# 3. For each response:
#    - Briefly acknowledge the answer
#    - Move to the next question
#    - When finishing a category, introduce the next one"""


# llm = AzureChatOpenAI(
#     azure_deployment="gpt-4o-mini",  # or your deployment
#     api_version="2024-02-15-preview",  # or your api version
#     temperature=0,
#     # other params...
#     )


# engine = create_engine("sqlite:///CompanyDB.db")

# db = SQLDatabase(engine=engine)
# print(db.dialect)
# print(db.get_usable_table_names())



# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# tools = toolkit.get_tools()

# system_message = SystemMessage(content=SQL_PREFIX)

# memory = MemorySaver()
    
# agent_executor = create_react_agent(llm, tools, messages_modifier=system_message,
#                                       checkpointer=memory)
    
# config = {"configurable": {"thread_id": "abc123"}}

# def initialize_session_state():
#     if 'user_id' not in st.session_state:
#         st.session_state.user_id = None
#     if 'conversation_active' not in st.session_state:
#         st.session_state.conversation_active = False
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
 
# def process_chat(agent_executor, query: str, is_first_message: bool = False):
#     if is_first_message:
#         # Create user in database only for the first message
#         user_response = requests.post(
#             "http://localhost:8000/users/",
#             json={"message": query}
#         ).json()   
#         st.session_state.user_id = user_response["user_id"]
#         st.session_state.conversation_active = True
    
#     messages = []
#     print(agent_executor.checkpointer.__dict__)
#     for s in agent_executor.stream(
        
#         {"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": "abc123"}}

#     ):      

#         if 'agent' in s and 'messages' in s['agent']:
#             message = s['agent']['messages'][0]
            
#             # If this is a question from the bot (not intermediate processing)
#             if message.content and not message.additional_kwargs.get('tool_calls'):
#                 # Save bot's message
#                 st.session_state.messages.append({"role": "assistant", "content": message.content})
                
#                 # Save the response to database
#                 if not is_first_message:  # Only save responses after the first message
#                     response = requests.post(
#                         "http://localhost:8000/responses/",
#                         json={
#                             "user_id": st.session_state.user_id,
#                             "bot_message": message.content,
#                             "user_answer": query  # This is the user's previous answer
#                         }
#                     )
#                 return message.content
#     return None

# def main():
#     st.title("Technical Assessment Chatbot")
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Your existing Azure OpenAI and agent setup code should go here
#     # agent_executor = ... (your existing setup)
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Your message"):
#         # Display user message
#         with st.chat_message("user"):
#             st.write(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Process the message
#         is_first_message = not st.session_state.conversation_active
#         memory = MemorySaver()
    
#         agent_executor = create_react_agent(llm, tools, messages_modifier=system_message,
#                                       checkpointer=memory)
#         with st.chat_message("assistant"):
#             response = process_chat(agent_executor, prompt, is_first_message)
#             if response:
#                 st.write(response)

# if __name__ == "__main__":
#     main()


# from dotenv import load_dotenv
# import os
# import streamlit as st
# import requests
# import pandas as pd
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_community.utilities import SQLDatabase
# from sqlalchemy import create_engine
# from sqlalchemy import create_engine, inspect, text
# from langchain_core.messages import SystemMessage
# from langchain_community.agent_toolkits import SQLDatabaseToolkit

# # Load environment variables from .env file
# load_dotenv()

# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# # Add validation to check if variables exist
# if not api_key:
#     raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
# if not endpoint:
#     raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")

# # if "AZURE_OPENAI_API_KEY" not in os.environ:
# #     os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
# #         "Enter your AzureOpenAI API key: "
# #     )




# from langchain_openai import AzureChatOpenAI

# def setup_database(csv_file_path):
#     """Initialize database and return engine"""
#     df = pd.read_csv(csv_file_path)
#     engine = create_engine("sqlite:///CompanyDB_test.db")
#     df.to_sql("CompanyDBTable", engine, index=False)
    
#     return engine

# def get_llm():
#     """Initialize and return LLM"""
#     return AzureChatOpenAI(
#             model="gpt-4o-mini",
#             azure_deployment="gpt-4o-mini",
#             api_version="2024-02-15-preview",
#             temperature=0,
#             api_key=api_key,
#             azure_endpoint=endpoint,
            
#         )

# SQL_PREFIX = """You are an intelligent assessment bot designed to conduct a maturity assessment for different departments. You'll guide users through a series of questions to evaluate their department's maturity level.

# Initial Greeting:
# When a user first connects, introduce yourself and explain:
# 1. The purpose of this assessment (evaluating department maturity)
# 2. What to expect (series of questions grouped by categories)
# 3. That their responses will be saved for analysis
# 4. Ask for their name, email, and department to begin

# Database Instructions:
# You have access to a SQLite database with a 'MaturityDBTable' table containing department-specific questions. Only ask questions from this database.

# IMPORTANT WORKFLOW:
# 1. First Interaction:
#    - Warmly greet the user and explain the process
#    - Extract their name, email, and department
#    - Use this SQL query to get their department's questions:
#      SELECT Category, Question FROM MaturityDBTable WHERE Department = ? ORDER BY Category;
#    - Store these questions for the conversation

# 2. Question Flow:
#    - Before starting questions, explain which categories will be covered
#    - Group questions by category
#    - Introduce each new category before its questions
#    - Ask one question at a time
#    - Wait for user response before proceeding
#    - Keep track of asked questions to avoid repetition

# 3. Response Handling:
#    - Acknowledge each answer briefly
#    - Transition smoothly to the next question
#    - When switching categories, announce the new category
#    - At the end, thank the user and explain that their responses have been recorded

# Remember to maintain a professional yet friendly tone throughout the assessment."""


# # Your existing setup code remains the same

# def initialize_session_state():
#     if 'user_id' not in st.session_state:
#         st.session_state.user_id = None
#     if 'conversation_active' not in st.session_state:
#         st.session_state.conversation_active = False
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     # Add these new session state variables
#     if 'memory' not in st.session_state:
#         st.session_state.memory = MemorySaver()
#     if 'agent_executor' not in st.session_state:
       
#             # Setup database
#             df = pd.read_csv("MaturityCheck.csv")
#             engine = create_engine("sqlite:///CompanyDB_test1.db")
#             df.to_sql("CompanyDBTable", engine, index=False)
#             print(df.columns.tolist())
   
            
#             # Initialize LLM
#             llm = get_llm()
            
#             # Setup database toolkit
#             db = SQLDatabase(engine=engine)
#             toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#             tools = toolkit.get_tools()
            
#             # Setup system message
#             system_message = SystemMessage(content=SQL_PREFIX)
            
#             # Create agent
#             st.session_state.agent_executor = create_react_agent(
#                 llm, 
#                 tools, 
#                 messages_modifier=system_message,
#                 checkpointer=st.session_state.memory
#             )
    
#     return True




# def process_chat(query: str, is_first_message: bool = False):
#     if is_first_message:
#         user_response = requests.post(
#             "http://localhost:8000/users/",
#             json={"message": query}
#         ).json()   
#         st.session_state.user_id = user_response["user_id"]
#         st.session_state.conversation_active = True
    
#     # Use the persistent agent_executor from session state
#     for s in st.session_state.agent_executor.stream(
#         {"messages": [HumanMessage(content=query)]}, 
#         config={"configurable": {"thread_id": "abc123"}}
#     ):      
#         if 'agent' in s and 'messages' in s['agent']:
#             message = s['agent']['messages'][0]
            
#             if message.content and not message.additional_kwargs.get('tool_calls'):
#                 st.session_state.messages.append({"role": "assistant", "content": message.content})
                
#                 if not is_first_message:
#                     response = requests.post(
#                         "http://localhost:8000/responses/",
#                         json={
#                             "user_id": st.session_state.user_id,
#                             "bot_message": message.content,
#                             "user_answer": query
#                         }
#                     )
#                 return message.content
#     return None

# def main():
#     st.title("Technical Assessment Chatbot")
    
#     # Initialize session state (including agent and memory)
#     initialize_session_state()
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Your message"):
#         with st.chat_message("user"):
#             st.write(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         is_first_message = not st.session_state.conversation_active
#         with st.chat_message("assistant"):
#             response = process_chat(prompt, is_first_message)
#             if response:
#                 st.write(response)

# if __name__ == "__main__":
#     main()










 # Working finr 


# from dotenv import load_dotenv
# import os
# import streamlit as st
# import requests
# import pandas as pd
# from langchain.schema import HumanMessage
# from langchain_openai import AzureChatOpenAI
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_community.utilities import SQLDatabase
# from sqlalchemy import create_engine, inspect, text
# from langchain_core.messages import SystemMessage
# from langchain_community.agent_toolkits import SQLDatabaseToolkit

# # Load environment variables from .env file
# load_dotenv()

# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# # Add validation to check if variables exist
# if not api_key:
#     raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
# if not endpoint:
#     raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")

# def get_llm():
#     """Initialize and return LLM"""
#     return AzureChatOpenAI(
#         model="gpt-4o-mini",
#         azure_deployment="gpt-4o-mini",
#         api_version="2024-02-15-preview",
#         temperature=0,
#         api_key=api_key,
#         azure_endpoint=endpoint,
#     )

# def is_connection_valid(engine):
#     """Check if database connection is still valid."""
#     try:
#         with engine.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         return True
#     except Exception:
#         return False

# def get_database_engine():
#     """Get or create database engine."""
#     # Check if engine exists in session state
#     if 'db_engine' not in st.session_state:
#         print("Creating new database engine...")
#         engine = create_engine("sqlite:///CompanyDB_test2.db")
        
#         # Check if table exists, create if it doesn't
#         inspector = inspect(engine)
#         if "CompanyDBTable" not in inspector.get_table_names():
#             df = pd.read_csv("MaturityCheck.csv")
#             df.to_sql("CompanyDBTable", engine, index=False)
#             print("Table created successfully")
#             print(df.columns.tolist())
        
#         # Store engine in session state
#         st.session_state.db_engine = engine
#         return engine
    
#     # If engine exists, verify connection is still valid
#     engine = st.session_state.db_engine
#     if is_connection_valid(engine):
#         print("Reusing existing database connection...")
#         return engine
#     else:
#         print("Connection lost, creating new engine...")
#         # Create new engine if connection is invalid
#         engine = create_engine("sqlite:///CompanyDB_test2.db")
#         st.session_state.db_engine = engine
#         return engine

# SQL_PREFIX = """You are an intelligent assessment bot designed to conduct a maturity assessment for different departments. You'll guide users through a series of questions to evaluate their department's maturity level.

# Initial Greeting:
# When a user first connects, introduce yourself and explain:
# 1. The purpose of this assessment (evaluating department maturity)
# 2. What to expect (series of questions grouped by categories)
# 3. That their responses will be saved for analysis
# 4. Ask for their name, email, and department to begin

# Database Instructions:
# You have access to a SQLite database with a 'MaturityDBTable' table containing department-specific questions. Only ask questions from this database.

# IMPORTANT WORKFLOW:
# 1. First Interaction:
#    - Warmly greet the user and explain the process
#    - Extract their name, email, and department
#    - Use this SQL query to get their department's questions:
#      SELECT Category, Question FROM MaturityDBTable WHERE Department = ? ORDER BY Category;
#    - Store these questions for the conversation

# 2. Question Flow:
#    - Before starting questions, explain which categories will be covered
#    - Group questions by category
#    - Introduce each new category before its questions
#    - Ask one question at a time
#    - Wait for user response before proceeding
#    - Keep track of asked questions to avoid repetition

# 3. Response Handling:
#    - Acknowledge each answer briefly
#    - Transition smoothly to the next question
#    - When switching categories, announce the new category
#    - At the end, thank the user and explain that their responses have been recorded

# Remember to maintain a professional yet friendly tone throughout the assessment."""

# def initialize_session_state():
#     """Initialize all session state variables and components."""
#     # Initialize basic session state variables
#     if 'user_id' not in st.session_state:
#         st.session_state.user_id = None
#     if 'conversation_active' not in st.session_state:
#         st.session_state.conversation_active = False
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'memory' not in st.session_state:
#         st.session_state.memory = MemorySaver()
    
#     # Initialize agent if not already present
#     if 'agent_executor' not in st.session_state:
#         try:
#             # Get or create database engine
#             engine = get_database_engine()
            
#             # Initialize LLM
#             llm = get_llm()
            
#             # Setup database toolkit
#             db = SQLDatabase(engine=engine)
#             toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#             tools = toolkit.get_tools()
            
#             # Setup system message
#             system_message = SystemMessage(content=SQL_PREFIX)
            
#             # Create agent
#             st.session_state.agent_executor = create_react_agent(
#                 llm, 
#                 tools, 
#                 messages_modifier=system_message,
#                 checkpointer=st.session_state.memory
#             )
            
#             return True
#         except Exception as e:
#             st.error(f"Error initializing session state: {str(e)}")
#             return False
    
#     return True

# def process_chat(query: str, is_first_message: bool = False):
#     """Process chat messages and handle API calls."""
#     if is_first_message:
#         user_response = requests.post(
#             "http://localhost:8000/users/",
#             json={"message": query}
#         ).json()   
#         st.session_state.user_id = user_response["user_id"]
#         st.session_state.conversation_active = True
    
#     # Use the persistent agent_executor from session state
#     for s in st.session_state.agent_executor.stream(
#         {"messages": [HumanMessage(content=query)]}, 
#         config={"configurable": {"thread_id": "abc123"}}
#     ):      
#         if 'agent' in s and 'messages' in s['agent']:
#             message = s['agent']['messages'][0]
            
#             if message.content and not message.additional_kwargs.get('tool_calls'):
#                 st.session_state.messages.append({"role": "assistant", "content": message.content})
                
#                 if not is_first_message:
#                     response = requests.post(
#                         "http://localhost:8000/responses/",
#                         json={
#                             "user_id": st.session_state.user_id,
#                             "bot_message": message.content,
#                             "user_answer": query
#                         }
#                     )
#                 return message.content
#     return None

# def main():
#     """Main application function."""
#     st.title("Technical Assessment Chatbot")
    
#     # Initialize session state (including agent and memory)
#     if not initialize_session_state():
#         st.error("Failed to initialize application. Please check the logs.")
#         return
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Your message"):
#         with st.chat_message("user"):
#             st.write(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         is_first_message = not st.session_state.conversation_active
#         with st.chat_message("assistant"):
#             response = process_chat(prompt, is_first_message)
#             if response:
#                 st.write(response)

# if __name__ == "__main__":
#     main()











from dotenv import load_dotenv
import os
import streamlit as st
import requests
import pandas as pd
# from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
from langchain_core.messages import SystemMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from sqlalchemy.orm import sessionmaker
from models import Base, CompanyDB 


from langchain_core.messages import SystemMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

from langchain.agents import AgentExecutor

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Add validation to check if variables exist
if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")




def get_llm():
    """Initialize and return LLM"""
    return AzureChatOpenAI(
        model="gpt-4o-mini",
        azure_deployment="gpt-4o-mini",
        api_version="2024-02-15-preview",
        temperature=0,
        api_key=api_key,
        azure_endpoint=endpoint,
    )

## updated code starts here 
def setup_database():
    """Setup database with proper initialization"""
    try:
        # Create engine with specific parameters for better connection handling
        engine = create_engine(
            "sqlite:///CompanyDB_test3.db",
            connect_args={"check_same_thread": False},  # Allow multiple threads
            pool_pre_ping=True,  # Check connection validity
            pool_recycle=3600    # Recycle connections after an hour
        )
        
        # Create all tables if they don't exist
        Base.metadata.create_all(engine)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if data needs to be loaded
        if session.query(CompanyDB).count() == 0:
            # Load data from CSV
            df = pd.read_csv("MaturityCheck.csv")
            CompanyDB.create_from_df(df, session)
            print("Database initialized with CSV data")
        
        session.close()
        return engine
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise


def get_database_engine():
    """Get or create database engine with proper connection handling"""
    if 'db_engine' not in st.session_state:
        st.session_state.db_engine = setup_database()
    
    engine = st.session_state.db_engine
    
    # Verify connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"Connection error: {e}")
        # Recreate engine if connection failed
        st.session_state.db_engine = setup_database()
        return st.session_state.db_engine


SQL_PREFIX = """You are an intelligent assessment bot designed to conduct a maturity assessment for different departments. You'll guide users through a series of questions to evaluate their department's maturity level.

Initial Greeting:
When a user first connects, introduce yourself and explain:
1. The purpose of this assessment (evaluating department maturity)
2. What to expect (series of questions grouped by categories)
3. That their responses will be saved for analysis
4. Ask for their name, email, and department to begin

Database Instructions:
You have access to a SQLite database with a 'MaturityDBTable' table containing department-specific questions. Only ask questions from this database.

IMPORTANT WORKFLOW:
1. First Interaction:
   - Warmly greet the user and explain the process
   - Extract their name, email, and department
   - Use this SQL query to get their department's questions:
     SELECT Category, Question FROM MaturityDBTable WHERE Department = ? ORDER BY Category;
   - Store these questions for the conversation

2. Question Flow:
   - Before starting questions, explain which categories will be covered
   - Group questions by category
   - Introduce each new category before its questions
   - Ask one question at a time
   - Wait for user response before proceeding
   - Keep track of asked questions to avoid repetition

3. Response Handling:
   - Acknowledge each answer briefly
   - Transition smoothly to the next question
   - When switching categories, announce the new category
   - At the end, thank the user and explain that their responses have been recorded

Remember to maintain a professional yet friendly tone throughout the assessment."""

system_message = SystemMessage(content=SQL_PREFIX)

# def initialize_session_state():
#     """Initialize session state with proper error handling"""
#     # Basic session state initialization
#     for key in ['user_id', 'conversation_active', 'messages', 'memory']:
#         if key not in st.session_state:
#             st.session_state[key] = None if key != 'messages' else []
    
#     if 'memory' not in st.session_state:
#         st.session_state.memory = MemorySaver()
    
#     if 'agent_executor' not in st.session_state:
#         try:
#             # Get database engine with proper connection handling
#             engine = get_database_engine()
            
#             # Initialize components
#             print("still working fine")
#             llm = get_llm()
#             db = SQLDatabase(engine=engine)
#             toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#             tools = toolkit.get_tools()
            
#             # Create agent using langgraph's create_react_agent with correct parameters
#             print("intilizinng llm to agent")
#             st.session_state.agent_executor = create_react_agent(
#                 llm, 
#                 tools, 
#                 messages_modifier=system_message,
#                 checkpointer=st.session_state.memory
#             )
#             return True
            
#         except Exception as e:
#             st.error(f"Failed to initialize application: {str(e)}")
#             return False
    
#     return True


def initialize_session_state():
    """Initialize session state with proper error handling"""
    # Initialize basic session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.conversation_active = False
        st.session_state.messages = []
        st.session_state.user_info = None
        st.session_state.memory = MemorySaver()
        st.session_state.current_category = None
        st.session_state.asked_questions = set()
    
    if not st.session_state.initialized:
        try:
            # Get database engine with proper connection handling
            engine = get_database_engine()
            
            # Initialize components
            llm = get_llm()
            db = SQLDatabase(engine=engine)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools = toolkit.get_tools()
            
            # Create agent using langgraph's create_react_agent
            st.session_state.agent_executor = create_react_agent(
                llm, 
                tools, 
                messages_modifier=system_message,
                checkpointer=st.session_state.memory
            )
            
            st.session_state.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            return False
    
    return True

# def process_chat(query: str, is_first_message: bool = False):
#     """Process chat messages and handle API calls."""
#     if is_first_message:
#         user_response = requests.post(
#             "http://localhost:8000/users/",
#             json={"message": query}
#         ).json()   
#         st.session_state.user_id = user_response["user_id"]
#         st.session_state.conversation_active = True
    
#     try:
#         # Stream the agent's response
#         for s in st.session_state.agent_executor.stream(
#             {"messages": [HumanMessage(content=query)]},
#             config={"configurable": {"thread_id": "abc123"}}
#         ):
#             if 'agent' in s and 'messages' in s['agent']:
#                 message = s['agent']['messages'][0]
                
#                 if message.content and not message.additional_kwargs.get('tool_calls'):
#                     st.session_state.messages.append({"role": "assistant", "content": message.content})
                    
#                     if not is_first_message:
#                         requests.post(
#                             "http://localhost:8000/responses/",
#                             json={
#                                 "user_id": st.session_state.user_id,
#                                 "bot_message": message.content,
#                                 "user_answer": query
#                             }
#                         )
#                     return message.content
#         return None
#     except Exception as e:
#         st.error(f"Error processing chat: {str(e)}")
#         return None
    

def process_chat(query: str, is_first_message: bool = False):
    """Process chat messages and handle API calls."""
    try:
        if is_first_message and not st.session_state.conversation_active:
            # Extract user information from the first message
            user_response = requests.post(
                "http://localhost:8000/users/",
                json={"message": query}
            ).json()   
            st.session_state.user_id = user_response["user_id"]
            st.session_state.conversation_active = True
            st.session_state.user_info = {
                "message": query,
                "processed": True
            }
        
        # Stream the agent's response
        for s in st.session_state.agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": st.session_state.user_id if hasattr(st.session_state, 'user_id') else "default"}}
        ):
            if 'agent' in s and 'messages' in s['agent']:
                message = s['agent']['messages'][0]
                
                if message.content and not message.additional_kwargs.get('tool_calls'):
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": message.content,
                        "timestamp": pd.Timestamp.now()
                    })
                    
                    if not is_first_message:
                        requests.post(
                            "http://localhost:8000/responses/",
                            json={
                                "user_id": st.session_state.user_id,
                                "bot_message": message.content,
                                "user_answer": query
                            }
                        )
                    return message.content
        return None
    except Exception as e:
        st.error(f"Error processing chat: {str(e)}")
        return None

# def main():
#     """Main application function."""
#     st.title("Technical Assessment Chatbot")
    
#     # Initialize session state (including agent and memory)
#     if not initialize_session_state():
#         st.error("Failed to initialize application. Please check the logs.")
#         return
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Your message"):
#         with st.chat_message("user"):
#             st.write(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         is_first_message = not st.session_state.conversation_active
#         with st.chat_message("assistant"):
#             response = process_chat(prompt, is_first_message)
#             if response:
#                 st.write(response)

# if __name__ == "__main__":
#     main()

def main():
    """Main application function."""
    st.title("Technical Assessment Chatbot")
    st.write("Welcome to the Technical Assessment Chatbot! Please provide the following information to proceed with your maturity assessment:")

# Instructions for required input fields
    st.write("- **Name**: Enter your full name.")
    st.write("- **Email**: Provide your email address to receive a summary of your assessment.")
    st.write("- **Department**: Let us know your department to help tailor the assessment.")

    
    # Initialize session state
    if not initialize_session_state():
        st.error("Failed to initialize application. Please check the logs.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Your message"):
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": pd.Timestamp.now()
        })
        
        # Process the message
        is_first_message = not st.session_state.conversation_active
        with st.chat_message("assistant"):
            response = process_chat(prompt, is_first_message)
            if response:
                st.write(response)

if __name__ == "__main__":
    main()