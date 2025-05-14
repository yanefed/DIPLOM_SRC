from datetime import datetime

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException

from app.database import get_db
from app.models.checklist import Checklist
from app.models.plane import Plane
from app.models.report import Report
from app.models.system import System
from app.permissions import has_permission

report_and_systems_router = APIRouter()


def create_trigger(db: Session):
    sql = """
    CREATE OR REPLACE FUNCTION decrease_k_coeff() RETURNS TRIGGER AS $$
    BEGIN
        UPDATE systems
        SET k_coeff = k_coeff - (random() * 10)
        WHERE plane = NEW.plane;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    
    CREATE OR REPLACE TRIGGER decrease_k_coeff_trigger
    AFTER INSERT ON reports
    FOR EACH ROW EXECUTE FUNCTION decrease_k_coeff();
    """
    db.execute(text(sql))


# [...] post all reports
@report_and_systems_router.post("/", status_code=status.HTTP_201_CREATED)
def create_reports_and_systems(db: Session = Depends(get_db)):
    try:
        create_trigger(db)
        db.commit()

        planes = db.query(Plane).all()
        if not planes:
            raise HTTPException(status_code=404, detail="No planes found in database")

        checklists = db.query(Checklist).all()
        if not checklists:
            raise HTTPException(status_code=404, detail="No checklists found in database")

        systems = db.query(System).all()
        if not systems:
            raise HTTPException(status_code=404, detail="No systems found in database")

        systems_to_check = [(system.name, system.category, system.k_coeff, system.plane) for system in systems]
        checklist_info = [(checklist.name, checklist.category) for checklist in checklists]

        successful_planes = []
        failed_planes = []

        for plane in planes:
            try:
                if not plane.tail_num:
                    failed_planes.append({"tail_num": None, "error": "No tail number"})
                    continue

                k_coeffs = [system[2] for system in systems_to_check if system[3] == plane.tail_num]
                if k_coeffs:
                    avg_rate = sum(k_coeffs) / len(k_coeffs)
                else:
                    avg_rate = 0

                # Decision is True if avg_rate is greater than 80, otherwise False
                decision = avg_rate > 80

                new_report = Report(
                    plane=plane.tail_num,
                    rate=avg_rate,
                    decision=decision,
                    check_time=datetime.utcnow()
                )
                db.add(new_report)
                db.flush()

                for checklist in checklists:
                    new_system = System(
                        plane=plane.tail_num,
                        name=checklist.name,
                        category=checklist.category,
                        k_coeff=100.0
                    )
                    db.add(new_system)

                db.commit()
                successful_planes.append(plane.tail_num)

            except Exception as e:
                db.rollback()
                failed_planes.append({"tail_num": plane.tail_num, "error": str(e)})
                continue

        return {
            "status": "success" if successful_planes else "warning",
            "message": "Reports and systems processing completed",
            "successful_planes": successful_planes,
            "failed_planes": failed_planes
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# [...] get all reports
@report_and_systems_router.get("/", status_code=status.HTTP_200_OK)
def read_reports_and_systems(db: Session = Depends(get_db)):
    reports = db.query(Report).all()
    systems = db.query(System).all()
    checklists = db.query(Checklist).all()

    reports_info = [
        {"plane": report.plane, "rate": report.rate, "decision": report.decision, "check_time": report.check_time} for
        report in reports]
    systems_info = [{"name": system.name, "category": system.category, "k_coeff": system.k_coeff, "plane": system.plane}
                    for system in systems]
    checklists_info = [{"name": checklist.name, "category": checklist.category} for checklist in checklists]

    return {"reports": reports_info, "systems": systems_info, "checklists": checklists_info}


# [...] delete all reports
@report_and_systems_router.delete("/", status_code=status.HTTP_200_OK)
def delete_reports_and_systems(db: Session = Depends(get_db)):
    db.query(Report).delete()
    db.query(System).delete()
    db.commit()
    return {"message": "Reports and systems deleted successfully"}


# [...] get reports by plane
@report_and_systems_router.get("/{plane}", status_code=status.HTTP_200_OK)
def read_reports_and_systems_by_plane(plane: str, db: Session = Depends(get_db)):
    reports = db.query(Report).filter(Report.plane == plane).all()
    systems = db.query(System).filter(System.plane == plane).all()
    checklists = db.query(Checklist).all()

    reports_info = [
        {"plane": report.plane, "rate": report.rate, "decision": report.decision, "check_time": report.check_time} for
        report in reports]
    systems_info = [{"name": system.name, "category": system.category, "k_coeff": system.k_coeff, "plane": system.plane}
                    for system in systems]
    checklists_info = [{"name": checklist.name, "category": checklist.category} for checklist in checklists]

    return {"reports": reports_info, "systems": systems_info, "checklists": checklists_info}


# [...] delete reports by plane
@report_and_systems_router.delete("/{plane}", status_code=status.HTTP_200_OK)
def delete_reports_and_systems_by_plane(plane: str, db: Session = Depends(get_db)):
    db.query(Report).filter(Report.plane == plane).delete()
    db.query(System).filter(System.plane == plane).delete()
    db.commit()
    return {"message": "Reports and systems deleted successfully"}


# [...] get reports by plane and category
@report_and_systems_router.get("/{plane}/{category}", status_code=status.HTTP_200_OK)
def read_reports_and_systems_by_plane_and_category(plane: str, category: str, db: Session = Depends(get_db)):
    reports = db.query(Report).filter(Report.plane == plane).all()
    systems = db.query(System).filter(System.plane == plane, System.category == category).all()
    checklists = db.query(Checklist).filter(Checklist.category == category).all()

    reports_info = [
        {"plane": report.plane, "rate": report.rate, "decision": report.decision, "check_time": report.check_time} for
        report in reports]
    systems_info = [{"name": system.name, "category": system.category, "k_coeff": system.k_coeff, "plane": system.plane}
                    for system in systems]
    checklists_info = [{"name": checklist.name, "category": checklist.category} for checklist in checklists]

    return {"reports": reports_info, "systems": systems_info, "checklists": checklists_info}


# [...] delete reports by plane and category
@report_and_systems_router.delete("/{plane}/{category}", status_code=status.HTTP_200_OK)
def delete_reports_and_systems_by_plane_and_category(plane: str, category: str, db: Session = Depends(get_db)):
    db.query(Report).filter(Report.plane == plane).delete()
    db.query(System).filter(System.plane == plane, System.category == category).delete()
    db.commit()
    return {"message": "Reports and systems deleted successfully"}


# [...] get reports by plane and category and name
@report_and_systems_router.get("/{plane}/{category}/{name}", status_code=status.HTTP_200_OK)
def read_reports_and_systems_by_plane_and_category_and_name(plane: str, category: str, name: str,
                                                            db: Session = Depends(get_db)):
    reports = db.query(Report).filter(Report.plane == plane).all()
    systems = db.query(System).filter(System.plane == plane, System.category == category, System.name == name).all()
    checklists = db.query(Checklist).filter(Checklist.category == category, Checklist.name == name).all()

    reports_info = [
        {"plane": report.plane, "rate": report.rate, "decision": report.decision, "check_time": report.check_time} for
        report in reports]
    systems_info = [{"name": system.name, "category": system.category, "k_coeff": system.k_coeff, "plane": system.plane}
                    for system in systems]
    checklists_info = [{"name": checklist.name, "category": checklist.category} for checklist in checklists]

    return {"reports": reports_info, "systems": systems_info, "checklists": checklists_info}


# [...] delete reports by plane and category and name
@report_and_systems_router.delete("/{plane}/{category}/{name}", status_code=status.HTTP_200_OK)
def delete_reports_and_systems_by_plane_and_category_and_name(plane: str, category: str, name: str,
                                                              db: Session = Depends(get_db)):
    db.query(Report).filter(Report.plane == plane).delete()
    db.query(System).filter(System.plane == plane, System.category == category, System.name == name).delete()
    db.commit()
    return {"message": "Reports and systems deleted successfully"}


# [...] get reports by plane and category and name and k_coeff
@report_and_systems_router.get("/{plane}/{category}/{name}/{k_coeff}", status_code=status.HTTP_200_OK)
def read_reports_and_systems_by_plane_and_category_and_name_and_k_coeff(plane: str, category: str, name: str,
                                                                        k_coeff: float, db: Session = Depends(get_db)):
    reports = db.query(Report).filter(Report.plane == plane).all()
    systems = db.query(System).filter(System.plane == plane, System.category == category, System.name == name,
                                      System.k_coeff == k_coeff).all()
    checklists = db.query(Checklist).filter(Checklist.category == category, Checklist.name == name).all()

    reports_info = [
        {"plane": report.plane, "rate": report.rate, "decision": report.decision, "check_time": report.check_time} for
        report in reports]
    systems_info = [{"name": system.name, "category": system.category, "k_coeff": system.k_coeff, "plane": system.plane}
                    for system in systems]
    checklists_info = [{"name": checklist.name, "category": checklist.category} for checklist in checklists]

    return {"reports": reports_info, "systems": systems_info, "checklists": checklists_info}


# [...] delete reports by plane and category and name and k_coeff
@report_and_systems_router.delete("/{plane}/{category}/{name}/{k_coeff}", status_code=status.HTTP_200_OK)
def delete_reports_and_systems_by_plane_and_category_and_name_and_k_coeff(plane: str, category: str, name: str,
                                                                          k_coeff: float,
                                                                          db: Session = Depends(get_db)):
    db.query(Report).filter(Report.plane == plane).delete()
    db.query(System).filter(System.plane == plane, System.category == category, System.name == name,
                            System.k_coeff == k_coeff).delete()
    db.commit()
    return {"message": "Reports and systems deleted successfully"}


# [...] post reports by plane and category and name and k_coeff
@report_and_systems_router.post("/{plane}/{category}/{name}/{k_coeff}", status_code=status.HTTP_201_CREATED)
def create_reports_and_systems_by_plane_and_category_and_name_and_k_coeff(plane: str, category: str, name: str,
                                                                          k_coeff: float,
                                                                          db: Session = Depends(get_db)):
    decision = k_coeff > 80  # Decision is True if avg_rate is greater than 80, otherwise False
    report = Report(plane=plane, rate=k_coeff, decision=decision, check_time=datetime.utcnow())
    db.add(report)
    db.commit()

    system = System(name=name, category=category, k_coeff=k_coeff, plane=plane)
    db.add(system)
    db.commit()

    return {"message": "Report and system created successfully"}
