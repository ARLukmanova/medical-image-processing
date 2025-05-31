from celery import Celery

celery_app = Celery("tasks", include=["worker.predict_task"])
