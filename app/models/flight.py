from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Flight(Base):
    __tablename__ = 'flights'

    id = Column(Integer, primary_key=True)
    fl_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    airline_code = Column(String, index=True)
    origin_airport = Column(String, index=True)
    dest_airport = Column(String, index=True)
    distance = Column(Integer)
    tail_num = Column(String, index=True)
    dep_time = Column(String, index=True)
    arr_time = Column(String, index=True)

    airports = relationship("FlightAirport", back_populates="flight")

    def __repr__(self):
        return f"<Flight {self.id}>"
