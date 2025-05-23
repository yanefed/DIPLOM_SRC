from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_HOSTNAME: str

    class Config:
        env_file = "./.env"


settings = Settings()
