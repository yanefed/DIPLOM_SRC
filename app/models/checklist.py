from sqlalchemy import Column, Integer, String

from app.database import Base


class Checklist(Base):
    __tablename__ = 'checklist'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, index=True)
    category = Column(String)

    def __repr__(self):
        return f"<Checklist {self.name}>"
