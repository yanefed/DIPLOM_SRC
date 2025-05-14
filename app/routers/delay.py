from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.delay import Delay
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.delay import DelayBaseSchema

delay_router = APIRouter()


# [...] get all delay records
@delay_router.get("/")
def get_delays(db: Session = Depends(get_db)):
    delays = db.query(Delay).all()
    return {"status": "ok", "message": "List of delays", "delays": delays}


# [...] add a new delay record
@delay_router.post("/", status_code=status.HTTP_201_CREATED)
def add_delay(delay: DelayBaseSchema, db: Session = Depends(get_db)):
    new_delay = Delay(
        dep_delay=delay.dep_delay,
        arr_delay=delay.arr_delay,
        cancelled=delay.cancelled,
        cancellation_code=delay.cancellation_code,
    )
    db.add(new_delay)
    db.commit()
    db.refresh(new_delay)
    return {"status": "ok", "message": "Delay added", "delay": new_delay}


# [...] edit a delay record
@delay_router.put("/{delay_id}")
def update_delay(delay_id: int, delay: DelayBaseSchema, db: Session = Depends(get_db)):
    delay_record = db.query(Delay).filter(Delay.id == delay_id).first()
    if not delay_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Delay not found")
    for key, value in delay.dict().items():
        setattr(delay_record, key, value)
    db.commit()
    db.refresh(delay_record)
    return {"status": "ok", "message": "Delay updated", "delay": delay_record}


# [...] get a single delay record
@delay_router.get("/{delay_id}")
def get_delay(delay_id: int, db: Session = Depends(get_db)):
    delay_record = db.query(Delay).filter(Delay.id == delay_id).first()
    if not delay_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Delay not found")
    return {"status": "ok", "message": "Delay found", "delay": delay_record}


# [...] delete a delay record
@delay_router.delete("/{delay_id}")
def delete_delay(delay_id: int, db: Session = Depends(get_db)):
    delay_record = db.query(Delay).filter(Delay.id == delay_id).first()
    if not delay_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Delay not found")
    db.delete(delay_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
