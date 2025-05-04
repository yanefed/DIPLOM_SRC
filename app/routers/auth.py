from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.users import current_users

auth_router = APIRouter()


# [...] authentication route
@auth_router.post("/")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    for user in current_users:
        if user["username"] == form_data.username and user["password"] == form_data.password:
            return {"access_token": user["password"], "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
