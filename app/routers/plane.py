from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.plane import Plane
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.plane import PlaneBaseSchema

plane_router = APIRouter()


# [...] get all plane records
@plane_router.get("/", dependencies=[Depends(has_permission("read:planes"))])
def get_planes(db: Session = Depends(get_db)):
    planes = db.query(Plane).all()
    return {"status": "ok", "message": "List of planes", "planes": planes}


# [...] add a new plane record
@plane_router.post("/", status_code=status.HTTP_201_CREATED, dependencies=[Depends(has_permission("write:planes"))])
def add_plane(plane: PlaneBaseSchema, db: Session = Depends(get_db)):
    new_plane = Plane(
        manufacture_year=plane.manufacture_year,
        tail_num=plane.tail_num,
        number_of_seats=plane.number_of_seats,
        plane_type=plane.type,
        airline_code=plane.airline_code,
    )
    db.add(new_plane)
    db.commit()
    db.refresh(new_plane)
    return {"status": "ok", "message": "Plane added", "plane": new_plane}


# [...] edit a plane record
@plane_router.put("/{plane_tail_num}", dependencies=[Depends(has_permission("write:planes"))])
def update_plane(plane_tail_num: str, plane: PlaneBaseSchema, db: Session = Depends(get_db)):
    plane_record = db.query(Plane).filter(Plane.tail_num == plane_tail_num).first()
    if not plane_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plane not found")
    for key, value in plane.dict().items():
        setattr(plane_record, key, value)
    db.commit()
    db.refresh(plane_record)
    return {"status": "ok", "message": "Plane updated", "plane": plane_record}


# [...] get a plane record
@plane_router.get("/{plane_tail_num}", dependencies=[Depends(has_permission("read:planes"))])
def get_plane(plane_tail_num: str, db: Session = Depends(get_db)):
    plane_record = db.query(Plane).filter(Plane.tail_num == plane_tail_num).first()
    if not plane_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plane not found")
    return {"status": "ok", "message": "Plane found", "plane": plane_record}


# [...] delete a plane record
@plane_router.delete("/{plane_tail_num}", dependencies=[Depends(has_permission("write:planes"))])
def delete_plane(plane_tail_num: str, db: Session = Depends(get_db)):
    plane_record = db.query(Plane).filter(Plane.tail_num == plane_tail_num).first()
    if not plane_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plane not found")
    db.delete(plane_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
