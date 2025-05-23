from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
from app.database import get_db
from app.models.airline import Airline
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.airline import AirlineBaseSchema

logger = logging.getLogger(__name__)

airline_router = APIRouter()


# [...] get all airline records
@airline_router.get("/")
def get_airlines(db: Session = Depends(get_db)):
    airlines = db.query(Airline).all()
    return {"status": "ok", "message": "List of airlines", "airlines": airlines}


# [...] get airlines by route
@airline_router.get("/route/{origin}/{destination}")
def get_airlines_by_route(origin: str, destination: str, db: Session = Depends(get_db)):
    """Get airlines that operate on a specific route."""
    # Логирование входных параметров
    logger.info(f"Received request for airlines on route from {origin} to {destination}")

    try:
        # SQL query to find airlines operating on the given route
        query = text("""
            SELECT DISTINCT a.airline_code, a.airline_name
            FROM flights f
            JOIN airlines a ON f.airline_code = a.airline_code
            WHERE UPPER(f.origin_airport) = UPPER(:origin)
            AND UPPER(f.dest_airport) = UPPER(:destination)
        """)

        # Лог SQL-запроса и параметров
        logger.info(f"Executing query with params: origin={origin.upper()}, destination={destination.upper()}")

        result = db.execute(query, {"origin": origin.upper(), "destination": destination.upper()})
        airlines = [{"airline_code": row[0], "airline_name": row[1]} for row in result]

        # Лог результатов
        logger.info(f"Found {len(airlines)} airlines for route {origin} to {destination}")

        # Если не найдено авиакомпаний для этого маршрута, добавим все авиакомпании
        if not airlines:
            logger.info(f"No airlines found for route {origin} to {destination}. Fetching all airlines.")
            all_airlines = db.query(Airline).all()
            airlines = [{"airline_code": a.airline_code, "airline_name": a.airline_name} for a in all_airlines]
            logger.info(f"Returning all {len(airlines)} airlines as fallback")

        return {
            "status": "ok",
            "message": f"Airlines operating from {origin} to {destination}",
            "airlines": airlines,
            "count": len(airlines)
        }
    except Exception as e:
        # Расширенное логирование ошибок
        logger.error(f"Error in get_airlines_by_route: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error retrieving airlines: {str(e)}",
            "airlines": []
        }


# [...] add a new airline record
@airline_router.post("/")
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
@airline_router.put("/{airline_code}")
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
@airline_router.get("/{airline_code}")
def get_airline(airline_code: str, db: Session = Depends(get_db)):
    airline_record = db.query(Airline).filter(Airline.airline_code == airline_code).first()
    if not airline_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airline not found")
    return {"status": "ok", "message": "Airline found", "airline": airline_record}


# [...] delete an airline record
@airline_router.delete("/{airline_code}")
def delete_airline(airline_code: str, db: Session = Depends(get_db)):
    airline_record = db.query(Airline).filter(Airline.airline_code == airline_code).first()
    if not airline_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Airline not found")
    db.delete(airline_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
