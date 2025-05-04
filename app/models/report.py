# app/models/report.py
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, TIMESTAMP, Sequence
from sqlalchemy.sql import func

from app.database import Base


class Report(Base):
    __tablename__ = 'reports'

    id_seq = Sequence('report_id_seq')
    id = Column(Integer,
                Sequence('report_id_seq'),
                primary_key=True,
                server_default=id_seq.next_value())
    plane = Column(String, ForeignKey('planes.tail_num'), nullable=False)
    check_time = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    rate = Column(Float, nullable=False, default=100.0)
    decision = Column(Boolean, nullable=False, default=True)

    def __repr__(self):
        return f"<Report {self.id}>"
