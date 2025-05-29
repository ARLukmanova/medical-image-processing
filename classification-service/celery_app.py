from celery import Celery

celery_app = Celery("tasks", include=["predict_task"])
