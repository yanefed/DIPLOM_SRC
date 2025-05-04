from fastapi import Depends, HTTPException, status

from app.users import current_users, oauth2_scheme, roles_permissions


def get_current_user(token: str = Depends(oauth2_scheme)):
    for user in current_users:
        if user["password"] == token:  # Используйте поле "password" в качестве токена
            return user
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found",
    )


def has_permission(permission: str):
    def _has_permission(current_user: dict = Depends(get_current_user)):
        user_roles = current_user["roles"]
        for role in user_roles:
            if permission in roles_permissions.get(role, []):
                return True
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    return _has_permission
