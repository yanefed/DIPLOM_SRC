from sqlalchemy import Column, Integer, String

from app.database import Base


class Crew(Base):
    __tablename__ = 'crew'

    tail_num = Column(String, index=True, primary_key=True)
    pilot1_name = Column(String)
    pilot1_rate = Column(Integer)
    pilot2_name = Column(String)
    steward1_name = Column(String)
    steward2_name = Column(String)
    steward3_name = Column(String)

    def __repr__(self):
        return f"<Crew {self.tail_num}>"
