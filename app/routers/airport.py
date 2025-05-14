from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.airport import Airport
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.airport import AirportBaseSchema

airport_router = APIRouter()


# [...] get all airport records
@airport_router.get("/")
def get_airports(db: Session = Depends(get_db)):
    airports = db.query(Airport).all()
    return {"status": "ok", "message": "List of airports", "airports": airports}


# [...] add a new airport record
@airport_router.post("/", status_code=status.HTTP_201_CREATED)
def add_airport(airport: AirportBaseSchema, db: Session = Depends(get_db)):
    new_airport = Airport(
        display_airport_name=airport.name,
        airport_code=airport.code,
        airport_city=airport.city,
        airport_fullname=airport.fullname,
        airport_state=airport.state,
        airport_country=airport.country,
        latitude=airport.lat,
        longitude=airport.lon
    )
    db.add(new_airport)
    db.commit()
    db.refresh(new_airport)
    return {"status": "ok", "message": "Airport added", "airport": new_airport}


# [...] edit an airport record
@airport_router.put("/{airport_code}")
def update_airport(airport_code: str, airport: AirportBaseSchema, db: Session = Depends(get_db)):
    airport_record = db.query(Airport).filter(Airport.airport_code == airport_code).first()
    if not airport_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airport not found")
    for key, value in airport.dict().items():
        setattr(airport_record, key, value)
    db.commit()
    return {"status": "ok", "message": "Airport updated", "airport": airport_record}


# [...] get a single airport record
@airport_router.get("/{airport_code}")
def get_airport(airport_code: str, db: Session = Depends(get_db)):
    airport = db.query(Airport).filter(Airport.airport_code == airport_code).first()
    if not airport:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airport not found")
    return {"status": "ok", "message": "Airport found", "airport": airport}


# [...] delete an airport record
@airport_router.delete("/{airport_code}")
def delete_airport(airport_code: str, db: Session = Depends(get_db)):
    airport = db.query(Airport).filter(Airport.airport_code == airport_code).first()
    if not airport:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airport not found")
    db.delete(airport)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
