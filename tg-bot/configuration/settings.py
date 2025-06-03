from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    prediction_service_url: str = 'unset'
    tg_bot_token: str = 'unset'

    def __init__(self, **values):
        super().__init__(**values)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_predict_endpoint(self):
        return self.prediction_service_url + "predict"

    def get_prediction_result_endpoint(self, task_id: str):
        return self.prediction_service_url + f"predict_status/{task_id}"


settings = Settings()
TIMEOUT = 30
RESULT_POLLING_MAX_RETRIES = 30
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
