from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import List
import json
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI


from dotenv import load_dotenv
import getpass
import os

# Initialize FastAPI app
app = FastAPI()




load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Add validation to check if variables exist
if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")


llm = AzureChatOpenAI(
            model="gpt-4o-mini",
            azure_deployment="gpt-4o-mini",
            api_version="2024-02-15-preview",
            temperature=0,
            api_key=api_key,
            azure_endpoint=endpoint,
            
        )

# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///MaturityDatabase.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    department = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    responses = relationship("Response", back_populates="user")

class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String)
    question = Column(String)
    answer = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="responses")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    message: str

class ResponseCreate(BaseModel):
    user_id: int
    bot_message: str  # The question from chatbot
    user_answer: str  # The user's answer

def extract_user_details(message: str) -> dict:
    prompt = f"""
    Extract the following information from the message:
    - name
    - email
    - department
    
    Message: {message}
    
    Return only a JSON object with these three fields.
    """
    
    response = llm.invoke(prompt)
    llm_content = response.content
    
    # Strip the code block markers if present
    if llm_content.startswith("```") and llm_content.endswith("```"):
        llm_content = llm_content[llm_content.find("{") : llm_content.rfind("}") + 1]

    # Parse the JSON
    try:
        return json.loads(llm_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def extract_category_and_question(bot_message: str) -> dict:
    prompt = f"""
    From the following chatbot message, extract:
    - category (e.g., Application, Data, Infrastructure)
    - question (the actual question being asked)
    
    Message: {bot_message}
    
    Return only a JSON object with these two fields.
    """
    
    # Call the LLM and get the AIMessage response
    response = llm.invoke(prompt)
    llm_content = response.content
    
    # Strip any code block markers if present (i.e., ```json ... ```)
    if llm_content.startswith("```") and llm_content.endswith("```"):
        llm_content = llm_content[llm_content.find("{") : llm_content.rfind("}") + 1]

    # Parse the JSON content
    try:
        return json.loads(llm_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    


@app.post("/users/")
async def create_user(user_input: UserCreate):
    db = SessionLocal()
    try:
        # llm = AzureChatOpenAI(
        #     azure_deployment="gpt-4o-mini",
        #     api_version="2024-02-15-preview",
        #     temperature=0
        # )
        
        user_details = extract_user_details(user_input.message)
        
        db_user = User(
            name=user_details["name"],
            email=user_details["email"],
            department=user_details["department"]
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        print(db_user.id)
        return {"user_id": db_user.id, "message": "User created successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.post("/responses/")
async def create_response(response: ResponseCreate):
    db = SessionLocal()
    try:
        # llm = AzureChatOpenAI(
        #     azure_deployment="gpt-4o-mini",
        #     api_version="2024-02-15-preview",
        #     temperature=0
        # )
        
        # Extract category and question from bot message
        extracted_data = extract_category_and_question(response.bot_message)
        
        db_response = Response(
            user_id=response.user_id,
            category=extracted_data["category"],
            question=extracted_data["question"],
            answer=response.user_answer
        )
        db.add(db_response)
        db.commit()
        db.refresh(db_response)
        
        return {"response_id": db_response.id, "message": "Response saved successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.get("/users/{user_id}/responses")
async def get_user_responses(user_id: int):
    db = SessionLocal()
    try:
        responses = db.query(Response).filter(Response.user_id == user_id).all()
        return responses
    finally:
        db.close()

