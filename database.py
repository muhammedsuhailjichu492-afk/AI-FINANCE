
import os
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "finance.db")

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    vendor = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    amount = Column(Float, nullable=False)
    category = Column(String(100), nullable=False, default="Uncategorized")
    source = Column(String(20), nullable=False, default="manual")  # ocr | csv | manual
    raw_text = Column(Text, nullable=True)
    document_path = Column(String(500), nullable=True)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, default=0.0)
    anomaly_reason = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Budget(Base):
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(100), unique=True, nullable=False)
    monthly_limit = Column(Float, nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(500), nullable=False)
    filepath = Column(String(500), nullable=False)
    extracted_text = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    transaction_id = Column(Integer, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
