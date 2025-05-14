from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.checklist import Checklist
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.checklist import ChecklistBaseSchema

checklist_router = APIRouter()


# [...] get all checklist records
@checklist_router.get("/")
def get_checklists(db: Session = Depends(get_db)):
    checklists = db.query(Checklist).all()
    return {"status": "ok", "message": "List of checklists", "checklists": checklists}


# [...] add a new checklist record
@checklist_router.post("/", status_code=status.HTTP_201_CREATED)
def add_checklist(checklist: ChecklistBaseSchema, db: Session = Depends(get_db)):
    new_checklist = Checklist(
        name=checklist.name,
        category=checklist.category,
    )
    db.add(new_checklist)
    db.commit()
    db.refresh(new_checklist)
    return {"status": "ok", "message": "Checklist added", "checklist": new_checklist}


# [...] edit a checklist record
@checklist_router.put("/{checklist_id}")
def update_checklist(checklist_id: int, checklist: ChecklistBaseSchema, db: Session = Depends(get_db)):
    checklist_record = db.query(Checklist).filter(Checklist.id == checklist_id).first()
    if not checklist_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")
    for key, value in checklist.dict().items():
        setattr(checklist_record, key, value)
    db.commit()
    db.refresh(checklist_record)
    return {"status": "ok", "message": "Checklist updated", "checklist": checklist_record}


# [...] get a single checklist record
@checklist_router.get("/{checklist_id}")
def get_checklist(checklist_id: int, db: Session = Depends(get_db)):
    checklist_record = db.query(Checklist).filter(Checklist.id == checklist_id).first()
    if not checklist_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")
    return {"status": "ok", "message": "Checklist found", "checklist": checklist_record}


# [...] delete a checklist record
@checklist_router.delete("/{checklist_id}")
def delete_checklist(checklist_id: int, db: Session = Depends(get_db)):
    checklist_record = db.query(Checklist).filter(Checklist.id == checklist_id).first()
    if not checklist_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Checklist not found")
    db.delete(checklist_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
