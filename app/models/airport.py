from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class Airport(Base):
    __tablename__ = 'airports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    display_airport_name = Column(String, index=True)
    airport_code = Column(String, index=True, primary_key=True)
    airport_city = Column(String, index=True)
    airport_fullname = Column(String, index=True)
    airport_state = Column(String, index=True)
    airport_country = Column(String, index=True)
    latitude = Column(Float)
    longitude = Column(Float)

    flights = relationship("FlightAirport", back_populates="airport")

    def __repr__(self):
        return f"<Airport {self.code}>"
