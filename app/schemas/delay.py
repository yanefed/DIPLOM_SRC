from typing import List

from pydantic import BaseModel


class DelayBaseSchema(BaseModel):
    id: int
    dep_delay: int
    arr_delay: int
    cancelled: int
    cancellation_code: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListDelayResponse(BaseModel):
    status: str
    message: str
    flights: List[DelayBaseSchema]
