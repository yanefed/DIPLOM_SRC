import uvicorn

from app.main import app

uvicorn.run(app, host="0.0.0.0", port=8000)
