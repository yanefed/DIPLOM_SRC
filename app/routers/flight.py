from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from ..database import get_db
from ..models.flight import Flight
from ..models.flight_airport import FlightAirport
from ..schemas.flight import FlightBaseSchema

flight_router = APIRouter()


# [...] get all flight records
@flight_router.get("/")
def get_flights(db: Session = Depends(get_db)):
    flights = db.query(Flight).all()
    return {"status": "ok", "message": "List of flights", "flights": flights}


# [...] add a new flight record
@flight_router.post("/", status_code=status.HTTP_201_CREATED)
def add_flight(flight: FlightBaseSchema, db: Session = Depends(get_db)):
    new_flight = Flight(fl_date=flight.fl_date, airline_code=flight.airline_code, distance=flight.distance,
                        tail_num=flight.tail_num, dep_time=flight.dep_time, arr_time=flight.arr_time)
    db.add(new_flight)
    for airport in flight.airports:
        new_flight_airport = FlightAirport(flight_id=new_flight.id, airport_id=airport.airport_id,
                                           airport_type=airport.airport_type)
        db.add(new_flight_airport)
    db.commit()
    db.refresh(new_flight)
    return {"status": "ok", "message": "Flight added", "flight": new_flight}


# [...] edit a flight record
@flight_router.put("/{flight_id}")
def update_flight(flight_id: int, flight: FlightBaseSchema, db: Session = Depends(get_db)):
    flight_record = db.query(Flight).filter(Flight.id == flight_id).first()
    if not flight_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flight not found")
    for key, value in flight.dict().items():
        setattr(flight_record, key, value)
    db.commit()
    db.refresh(flight_record)
    return {"status": "ok", "message": "Flight updated", "flight": flight_record}


# [...] get a single flight record
@flight_router.get("/{flight_id}")
def get_flight(flight_id: int, db: Session = Depends(get_db)):
    flight_record = db.query(Flight).filter(Flight.id == flight_id).first()
    if not flight_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flight not found")
    return {"status": "ok", "message": "Flight details", "flight": flight_record}


# [...] delete a flight record
@flight_router.delete("/{flight_id}")
def delete_flight(flight_id: int, db: Session = Depends(get_db)):
    flight_record = db.query(Flight).filter(Flight.id == flight_id).first()
    if not flight_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flight not found")
    db.delete(flight_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
