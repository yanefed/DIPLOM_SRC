from typing import List

from pydantic import BaseModel


class PlaneBaseSchema(BaseModel):
    manufacture_year: int
    tail_num: str
    number_of_seats: int
    type: str
    airline_code: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListPlaneResponse(BaseModel):
    status: str
    message: str
    planes: List[PlaneBaseSchema]
