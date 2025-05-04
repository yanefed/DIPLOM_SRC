from typing import List

from pydantic import BaseModel


class CrewBaseSchema(BaseModel):
    tail_num: str
    pilot1_name: str
    pilot1_rate: int
    pilot2_name: str
    steward1_name: str
    steward2_name: str
    steward3_name: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListCrewResponse(BaseModel):
    status: str
    message: str
    crews: List[CrewBaseSchema]
