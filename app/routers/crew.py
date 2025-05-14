from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.crew import Crew
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.crew import CrewBaseSchema

crew_router = APIRouter()


# [...] get all crew records
@crew_router.get("/")
def get_crews(db: Session = Depends(get_db)):
    crews = db.query(Crew).all()
    return {"status": "ok", "message": "List of crews", "crews": crews}


# [...] add a new crew record
@crew_router.post("/", status_code=status.HTTP_201_CREATED)
def add_crew(crew: CrewBaseSchema, db: Session = Depends(get_db)):
    new_crew = Crew(
        tail_num=crew.tail_num,
        pilot1_name=crew.pilot1_name,
        pilot1_rate=crew.pilot1_rate,
        pilot2_name=crew.pilot2_name,
        steward1_name=crew.steward1_name,
        steward2_name=crew.steward2_name,
        steward3_name=crew.steward3_name,
    )
    db.add(new_crew)
    db.commit()
    db.refresh(new_crew)
    return {"status": "ok", "message": "Crew added", "crew": new_crew}


# [...] edit a crew record
@crew_router.put("/{crew_tail_num}")
def update_crew(crew_tail_num: str, crew: CrewBaseSchema, db: Session = Depends(get_db)):
    crew_record = db.query(Crew).filter(Crew.tail_num == crew_tail_num).first()
    if not crew_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crew not found")
    for key, value in crew.dict().items():
        setattr(crew_record, key, value)
    db.commit()
    db.refresh(crew_record)
    return {"status": "ok", "message": "Crew updated", "crew": crew_record}


# [...] get a single crew record
@crew_router.get("/{crew_tail_num}")
def get_crew(crew_tail_num: str, db: Session = Depends(get_db)):
    crew_record = db.query(Crew).filter(Crew.tail_num == crew_tail_num).first()
    if not crew_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crew not found")
    return {"status": "ok", "message": "Crew found", "crew": crew_record}


# [...] delete a crew record
@crew_router.delete("/{crew_tail_num}")
def delete_crew(crew_tail_num: str, db: Session = Depends(get_db)):
    crew_record = db.query(Crew).filter(Crew.tail_num == crew_tail_num).first()
    if not crew_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crew not found")
    db.delete(crew_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
