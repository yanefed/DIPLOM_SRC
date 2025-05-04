from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.airline import Airline
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.airline import AirlineBaseSchema

airline_router = APIRouter()


# [...] get all airline records
@airline_router.get("/", dependencies=[Depends(has_permission("read:airlines"))])
def get_airlines(db: Session = Depends(get_db)):
    airlines = db.query(Airline).all()
    return {"status": "ok", "message": "List of airlines", "airlines": airlines}


# [...] add a new airline record
@airline_router.post("/", status_code=status.HTTP_201_CREATED, dependencies=[Depends(has_permission("write:airlines"))])
def add_airline(airline: AirlineBaseSchema, db: Session = Depends(get_db)):
    new_airline = Airline(
        airline_name=airline.name,
        airline_code=airline.code,
    )
    db.add(new_airline)
    db.commit()
    db.refresh(new_airline)
    return {"status": "ok", "message": "Airline added", "airline": new_airline}


# [...] edit an airline record
@airline_router.put("/{airline_code}", dependencies=[Depends(has_permission("write:airlines"))])
def update_airline(airline_code: str, airline: AirlineBaseSchema, db: Session = Depends(get_db)):
    airline_record = db.query(Airline).filter(Airline.airline_code == airline_code).first()
    if not airline_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airline not found")
    for key, value in airline.dict().items():
        setattr(airline_record, key, value)
    db.commit()
    db.refresh(airline_record)
    return {"status": "ok", "message": "Airline updated", "airline": airline_record}


# [...] get a single airline record
@airline_router.get("/{airline_code}", dependencies=[Depends(has_permission("read:airlines"))])
def get_airline(airline_code: str, db: Session = Depends(get_db)):
    airline_record = db.query(Airline).filter(Airline.airline_code == airline_code).first()
    if not airline_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airline not found")
    return {"status": "ok", "message": "Airline found", "airline": airline_record}


# [...] delete an airline record
@airline_router.delete("/{airline_code}", dependencies=[Depends(has_permission("write:airlines"))])
def delete_airline(airline_code: str, db: Session = Depends(get_db)):
    airline_record = db.query(Airline).filter(Airline.airline_code == airline_code).first()
    if not airline_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airline not found")
    db.delete(airline_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
