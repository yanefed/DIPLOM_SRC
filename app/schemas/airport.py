from typing import List

from pydantic import BaseModel


class AirportBaseSchema(BaseModel):
    id: int
    code: str
    name: str
    city: str
    fullname: str
    state: str
    country: str
    lat: float
    lon: float

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListAirportResponse(BaseModel):
    status: str
    message: str
    airports: List[AirportBaseSchema]
