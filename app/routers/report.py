from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.report import Report
from app.permissions import has_permission  # Импортируйте функцию has_permission из main.py
from app.schemas.report import ReportBaseSchema

report_router = APIRouter()


# [...] get all report records
@report_router.get("/")
def get_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    return {"status": "ok", "message": "List of reports", "reports": reports}


# [...] add a new report record
@report_router.post("/", status_code=status.HTTP_201_CREATED)
def add_report(report: ReportBaseSchema, db: Session = Depends(get_db)):
    new_report = Report(plane=report.plane, check_time=report.check_time, rate=report.rate, decision=report.decision, )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    return {"status": "ok", "message": "Report added", "report": new_report}


# [...] edit a report record
@report_router.put("/{report_id}")
def update_report(report_id: int, report: ReportBaseSchema, db: Session = Depends(get_db)):
    report_record = db.query(Report).filter(Report.id == report_id).first()
    if not report_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    for key, value in report.dict().items():
        setattr(report_record, key, value)
    db.commit()
    db.refresh(report_record)
    return {"status": "ok", "message": "Report updated", "report": report_record}


# [...] get a report record
@report_router.get("/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    report_record = db.query(Report).filter(Report.id == report_id).first()
    if not report_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return {"status": "ok", "message": "Report found", "report": report_record}


# [...] delete a report record
@report_router.delete("/{report_id}")
def delete_report(report_id: int, db: Session = Depends(get_db)):
    report_record = db.query(Report).filter(Report.id == report_id).first()
    if not report_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    db.delete(report_record)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
