from celery_app import celery_app
from worker.model import Model
from worker.parameters import CLASS_NAMES
from worker.plot_tools import create_probability_plot
from worker.s3_tools import save_image_to_s3

model = Model()


@celery_app.task
def predict_task(image_bytes, filename):
    pred = model.predict(image_bytes)
    plot_base64 = create_probability_plot(pred['probabilities'])
    save_image_to_s3(image_bytes, filename)

    pneumonia_prob = pred['probabilities'][1]  # Вероятность пневмонии
    is_pneumonia = pred['prediction'] == 1
    return {
        "status": "success",
        "filename": filename,
        "prediction": {
            "class": CLASS_NAMES[pred['prediction']],
            "probability": float(pred['probability']),
            "is_pneumonia": bool(is_pneumonia),
            "pneumonia_probability": float(pneumonia_prob),
            "normal_probability": float(pred['probabilities'][0]),
            "message": "Анализ выполнен гибридной моделью (CNN + ViT)"
        },
        "plot_image": plot_base64,
        "recommendation": {
            "action": "Рекомендуется консультация врача" if is_pneumonia else "Патологий не обнаружено",
            "urgency": "Срочно" if pneumonia_prob > 0.7 else "Планово" if pneumonia_prob > 0.3 else "Не требуется"
        }
    }
