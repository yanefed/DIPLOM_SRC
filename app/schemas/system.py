from typing import List

from pydantic import BaseModel


class SystemBaseSchema(BaseModel):
    id: int
    plane: str
    name: str
    category: str
    k_coeff: float

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListSystemResponse(BaseModel):
    status: str
    message: str
    flights: List[SystemBaseSchema]
