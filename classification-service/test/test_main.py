from fastapi.testclient import TestClient
from api.app import app
from celery_app import celery_app

client = TestClient(app)


def test_predict_status_pending(monkeypatch):
    class DummyTask:
        state = "PENDING"
        id = "123"

    monkeypatch.setattr(celery_app, 'AsyncResult', lambda tid: DummyTask())
    response = client.get("/predict_status/123")
    assert response.status_code == 202
    assert response.json()['status'] == 'processing'
