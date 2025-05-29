import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from celery_app import celery_app
from logger import logger

app = FastAPI(title="X-Ray Classification API")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Конечная точка для классификации рентгеновских снимков"""
    try:
        logger.info(f"Получен запрос на классификацию от {file.filename}")

        file_bytes = await file.read()
        task = celery_app.send_task("predict_task.predict_task", args=[file_bytes, file.filename])
        return JSONResponse(content={
            "task_id": task.id,
            "status": "processing"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Необработанная ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/predict_status/{task_id}")
async def get_prediction_status(task_id: str) -> JSONResponse:
    task = celery_app.AsyncResult(task_id)
    if task.state == "PENDING":
        result =  {"status": "processing"}
    elif task.state == "SUCCESS":
        result = {"status": "done", "result": task.result}
    elif task.state == "FAILURE":
        result =  {"status": "error", "error": str(task.info)}
    else:
        result =  {"status": task.state}
    return JSONResponse(content=result)

# @app.get("/update_model_version")
# async def update_model_version_endpoint() -> JSONResponse:
#     try:
#         logger.info("Запрос на обновление версии модели")
#         download_new_model_version()
#         model.load_model()
#         return JSONResponse(content={
#             "status": "success",
#             "message": "Модель успешно обновлена"
#         })
#     except Exception as e:
#         logger.error(f"Ошибка при обновлении модели: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Ошибка при обновлении модели")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Запуск сервера на http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
