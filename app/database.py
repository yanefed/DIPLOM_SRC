import logging

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings

POSTGRES_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOSTNAME}:{settings.DATABASE_PORT}/{settings.POSTGRES_DB}"

logging.basicConfig(
    filename='sql_debug.log',  # имя файла для логов
    filemode='a',             # режим append - добавление записей в конец файла
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.DEBUG)
logging.getLogger('sqlalchemy.pool').setLevel(logging.DEBUG)
logging.getLogger('sqlalchemy.orm').setLevel(logging.DEBUG)

engine = create_engine(POSTGRES_URL, echo=True)

with engine.connect() as connection:
    connection.execute(text('create EXTENSION if not EXISTS "pgcrypto"'))
    connection.commit()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
