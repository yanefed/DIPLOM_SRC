from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.system import System
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.system import SystemBaseSchema

system_router = APIRouter()


# [...] get all system records
@system_router.get("/")
def get_systems(db: Session = Depends(get_db)):
    systems = db.query(System).all()
    return {"status": "ok", "message": "List of systems", "systems": systems}


# [...] add a new system record
@system_router.post("/", status_code=status.HTTP_201_CREATED)
def add_system(system: SystemBaseSchema, db: Session = Depends(get_db)):
    new_system = System(
        plane=system.plane,
        name=system.name,
        category=system.category,
        k_coeff=system.k_coeff,
    )
    db.add(new_system)
    db.commit()
    db.refresh(new_system)
    return {"status": "ok", "message": "System added", "system": new_system}


# [...] edit a system record
@system_router.put("/{system_id}")
def update_system(system_id: int, system: SystemBaseSchema, db: Session = Depends(get_db)):
    system_record = db.query(System).filter(System.id == system_id).first()
    if not system_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="System not found")
    for key, value in system.dict().items():
        setattr(system_record, key, value)
    db.commit()
    db.refresh(system_record)
    return {"status": "ok", "message": "System updated", "system": system_record}


# [...] get a system record
@system_router.get("/{system_id}")
def get_system(system_id: int, db: Session = Depends(get_db)):
    system_record = db.query(System).filter(System.id == system_id).first()
    if not system_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="System not found")
    return {"status": "ok", "message": "System found", "system": system_record}


# [...] delete a system record
@system_router.delete("/{system_id}")
def delete_system(system_id: int, db: Session = Depends(get_db)):
    system_record = db.query(System).filter(System.id == system_id).first()
    if not system_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="System not found")
    db.delete(system_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
