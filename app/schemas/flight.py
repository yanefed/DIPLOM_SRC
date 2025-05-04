from datetime import datetime
from typing import List

from pydantic import BaseModel


class FlightAirportSchema(BaseModel):
    flight_id: int
    airport_id: int
    airport_type: str

    class Config:
        orm_mode = True


class FlightBaseSchema(BaseModel):
    id: int
    fl_date: datetime
    airline_code: str
    origin_airport_code: str
    dest_airport_code: str
    distance: int
    tail_num: str
    dep_time: str
    arr_time: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListFlightResponse(BaseModel):
    status: str
    message: str
    flights: List[FlightBaseSchema]
