from typing import List

from pydantic import BaseModel


class AirlineBaseSchema(BaseModel):
    name: str
    code: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListAirlineResponse(BaseModel):
    status: str
    message: str
    airlines: List[AirlineBaseSchema]
