from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

roles_permissions = {
    "guest"   : ["read:flights", "read:airports", "read:airlines", "read:delays"],
    "employee": ["read:reports", "write:reports", "read:systems", "write:systems", "read:checklist", "write:checklist",
                 "read:reports_and_systems", "write:reports_and_systems"],
    "admin"   : ["read:flights", "write:flights", "read:airports", "write:airports",
                 "read:airlines", "write:airlines", "read:delays", "write:delays",
                 "read:reports", "write:reports", "read:systems", "write:systems",
                 "read:checklist", "write:checklist", "read:crew", "write:crew",
                 "read:reports_and_systems", "write:reports_and_systems"]
}

current_users = [
    {"username": "admin", "roles": ["admin"], "password": "admin"},
    {"username": "employee", "roles": ["employee"], "password": "employee"},
    {"username": "guest", "roles": ["guest"], "password": "guest"},
]
