from sqlalchemy import Column, Integer, String

from app.database import Base


class Plane(Base):
    __tablename__ = 'planes'

    manufacture_year = Column(Integer)
    tail_num = Column(String, index=True, primary_key=True)
    number_of_seats = Column(Integer)
    plane_type = Column(String, index=True)
    airline_code = Column(String, index=True)

    def __repr__(self):
        return f"<Plane {self.tail_num}>"
