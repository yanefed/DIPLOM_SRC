from sqlalchemy import Column, Float, ForeignKey, Integer, String, Sequence

from app.database import Base


class System(Base):
    __tablename__ = 'systems'

    id_seq = Sequence('report_id_seq')
    id = Column(Integer,
                Sequence('report_id_seq'),
                primary_key=True,
                server_default=id_seq.next_value())
    plane = Column(String, ForeignKey('planes.tail_num'), nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    k_coeff = Column(Float, nullable=False, default=100.0)

    def __repr__(self):
        return f"<System {self.id}>"
