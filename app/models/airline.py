from sqlalchemy import Column, String

from app.database import Base


class Airline(Base):
    __tablename__ = 'airlines'

    airline_name = Column(String, index=True)
    airline_code = Column(String, index=True, primary_key=True)

    def __repr__(self):
        return f"<Airline {self.name}>"
