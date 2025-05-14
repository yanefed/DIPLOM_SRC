from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from app.routers import airline_router, airport_router, auth_router, checklist_router, crew_router, delay_router, \
    flight_router, plane_router, probability_router, report_and_systems_router, report_router, system_router, delay_prediction_router

# START COMMAND:
# uvicorn app.main:app --reload

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()
app.include_router(flight_router, prefix="/api/v1/flights", tags=["flights"])
app.include_router(airport_router, prefix="/api/v1/airports", tags=["airports"])
app.include_router(airline_router, prefix="/api/v1/airlines", tags=["airlines"])
app.include_router(plane_router, prefix="/api/v1/planes", tags=["planes"])
app.include_router(crew_router, prefix="/api/v1/crew", tags=["crew"])
app.include_router(delay_router, prefix="/api/v1/delay", tags=["delay"])
app.include_router(system_router, prefix="/api/v1/system", tags=["system"])
app.include_router(report_router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(report_and_systems_router, prefix="/api/v1/report_and_systems", tags=["report_and_systems"])
app.include_router(checklist_router, prefix="/api/v1/checklists", tags=["checklist"])
app.include_router(probability_router, prefix="/api/v1/probability", tags=["probability"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(delay_prediction_router, prefix="/api/v1/predict", tags=["prediction"])

@app.get("/")
async def read_root():
    return FileResponse("app/static/index.html")
