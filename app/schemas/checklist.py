from typing import List

from pydantic import BaseModel


class ChecklistBaseSchema(BaseModel):
    id: int
    name: str
    category: str

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListChecklistResponse(BaseModel):
    status: str
    message: str
    checklists: List[ChecklistBaseSchema]
