from datetime import datetime
from typing import List

from pydantic import BaseModel


class ReportBaseSchema(BaseModel):
    id: int
    plane: str
    check_time: datetime
    rate: float
    decision: bool

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListReportResponse(BaseModel):
    status: str
    message: str
    flights: List[ReportBaseSchema]
