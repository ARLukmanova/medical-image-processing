from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    prediction_service_url: str
    tg_bot_token: str

    def __init__(self, **values):
        super().__init__(**values)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
