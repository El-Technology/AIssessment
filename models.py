# models.py
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class CompanyDB(Base):
    __tablename__ = 'CompanyDBTable'
    
    id = Column(Integer, primary_key=True)
    Department = Column(String)
    Category = Column(String)
    Question = Column(String)
    
    @classmethod
    def create_from_df(cls, df, session):
        """Create database entries from DataFrame"""
        for _, row in df.iterrows():
            record = cls(**row.to_dict())
            session.add(record)
        session.commit()