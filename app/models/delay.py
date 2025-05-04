from sqlalchemy import Column, Integer, String

from app.database import Base


class Delay(Base):
    __tablename__ = 'delay'

    id = Column(Integer, primary_key=True)
    dep_delay = Column(Integer)
    arr_delay = Column(Integer)
    cancelled = Column(Integer)
    cancellation_code = Column(String)

    def __repr__(self):
        return f"<Delay {self.id}>"
