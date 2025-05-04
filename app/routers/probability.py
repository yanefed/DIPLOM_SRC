from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db

probability_router = APIRouter()


def probability(db: Session, plane: str, origin: str, destination: str, date: str):
    # Call the stored procedure
    db.execute(
        text("CALL probability(:plane, :origin, :destination, :date)"),
        {"plane": plane, "origin": origin, "destination": destination, "date": date}
    )
    # Select from the temporary table
    result = db.execute(text("SELECT * FROM temp_result LIMIT 1"))
    return result.fetchone()


@probability_router.get("/{plane}/{origin}/{destination}/{date}")
# /N208WN/SJC/LAX/2024-05-05
def get_probability(plane: str, origin: str, destination: str, date: str, db: Session = Depends(get_db)):
    result = probability(db, plane, origin, destination, date)
    if not result:
        raise HTTPException(status_code=404, detail="No data found for the given parameters.")
    return {
        "airport"         : result[0],
        "average_delay"   : result[1],
        "total_delays"    : result[2],
        "delay_percentage": result[3]
    }
