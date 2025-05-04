from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class FlightAirport(Base):
    __tablename__ = 'flight_airport'

    flight_id = Column(Integer, ForeignKey('flights.id'), primary_key=True)
    airport_id = Column(Integer, ForeignKey('airports.id'), primary_key=True)
    airport_type = Column(String, nullable=False)  # 'departure' or 'arrival'

    flight = relationship("Flight", back_populates="airports")
    airport = relationship("Airport", back_populates="flights")
