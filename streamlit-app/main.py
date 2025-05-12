import io

import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image

from configuration.settings import settings

# Настройки
FASTAPI_URL = settings.prediction_service_url
CLASS_NAMES = ['Норма', 'Пневмония']

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ рентгеновских снимков",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS для красивого оформления
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .diagnosis {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .pneumonia {
        color: #e74c3c;
    }
    .normal {
        color: #2ecc71;
    }
    .probability {
        font-size: 20px;
        margin-bottom: 15px;
    }
    .model-result {
        padding: 10px;
        border-left: 4px solid #3498db;
        margin-bottom: 8px;
        background-color: #f8f9fa;
    }
    .recommendation {
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .warning {
        background-color: #fde8e8;
        border-left: 4px solid #e74c3c;
    }
    .success {
        background-color: #e8f8f0;
        border-left: 4px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown("<h1 class='title'>Анализ рентгеновских снимков на пневмонию</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        Загрузите рентгеновский снимок грудной клетки, и мы определим, есть ли признаки пневмонии.
    </div>
""", unsafe_allow_html=True)


# Функция для отправки изображения на сервер
def predict_image(image_bytes):
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(FASTAPI_URL, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка сервера: {response.json().get('detail', 'Неизвестная ошибка')}")
            return None
    except Exception as e:
        st.error(f"Ошибка при отправке запроса: {str(e)}")
        return None


# Функция для создания графика
def create_probability_plot(models_data):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Фильтруем модели с ошибками
    valid_models = {k: v for k, v in models_data.items() if 'probabilities' in v}

    if not valid_models:
        return None

    # Данные для графика
    models = [pred['name'] for pred in valid_models.values()]
    pneumonia_probs = [data['probabilities'][1] for data in valid_models.values()]
    normal_probs = [data['probabilities'][0] for data in valid_models.values()]

    # Среднее значение
    avg_pneumonia = sum(pneumonia_probs) / len(pneumonia_probs)

    # Построение графика
    x = range(len(models))
    width = 0.35

    ax.bar(x, normal_probs, width, label='Норма', color='#2ecc71')
    ax.bar(x, pneumonia_probs, width, bottom=normal_probs, label='Пневмония', color='#e74c3c')

    # Линия среднего значения
    ax.axhline(y=avg_pneumonia, color='#3498db', linestyle='--',
               label=f'Средняя вероятность пневмонии: {avg_pneumonia:.1%}')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Вероятность')
    ax.set_title('Результаты классификации разными моделями')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    return fig


# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите рентгеновский снимок (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    help="Загрузите рентгеновский снимок грудной клетки для анализа"
)

if uploaded_file is not None:
    # Показ загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Ваш рентгеновский снимок", width=300)

    # Конвертация в bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Кнопка для анализа
    if st.button("Анализировать снимок", type="primary"):
        with st.spinner("Анализируем снимок..."):
            # Отправка на сервер
            result = predict_image(img_bytes)

            if result:
                # Отображение результатов
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)

                # Итоговый результат
                final_pred = result["final_prediction"]
                diagnosis_class = "pneumonia" if final_pred["class"] == "Пневмония" else "normal"

                st.markdown(f"""
                    <div class='diagnosis {diagnosis_class}'>
                        ЗАКЛЮЧЕНИЕ: {final_pred["class"].upper()}
                    </div>
                    <div class='probability'>
                        Вероятность: <strong>{final_pred['probability'] * 100:.1f}%</strong>
                    </div>
                """, unsafe_allow_html=True)

                # Результаты по моделям
                st.markdown("<h4>Результаты по моделям:</h4>", unsafe_allow_html=True)
                for model, pred in result["models_predictions"].items():
                    st.markdown(f"""
                        <div class='model-result'>
                            <b>{pred['name']}</b>: {pred['class']} ({pred['probability'] * 100:.1f}%)
                        </div>
                    """, unsafe_allow_html=True)

                # Рекомендации
                rec_class = "warning" if final_pred["class"] == "Пневмония" else "success"
                rec_text = ("❗️ Требуется консультация врача!"
                            if final_pred["class"] == "Пневмония"
                            else "✅ Патологий не обнаружено")

                st.markdown(f"""
                    <div class='recommendation {rec_class}'>
                        <h4>РЕКОМЕНДАЦИИ</h4>
                        {rec_text}
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # График вероятностей
                fig = create_probability_plot(result["models_predictions"])
                if fig:
                    st.pyplot(fig)

                # Интерпретация результатов
                with st.expander("Как интерпретировать результаты"):
                    st.markdown("""
                        - **Вероятность до 20%** - скорее всего норма
                        - **20-60%** - рекомендуется повторный анализ
                        - **60-90%** - высокая вероятность патологии
                        - **Выше 90%** - срочно к врачу

                        Разные модели могут давать разные оценки. 
                        Средний результат считается как среднее арифметическое всех моделей.
                    """)

# Информация о приложении в сайдбаре
st.sidebar.markdown("""
### О приложении
Это приложение использует набор нейросетей для анализа рентгеновских снимков грудной клетки на признаки пневмонии.

**Используемые модели:**
- ResNet18
- EfficientNet
- VGG16
- ViT
- Hybrid

**Как использовать:**
1. Загрузите рентгеновский снимок
2. Нажмите кнопку "Анализировать снимок"
3. Получите подробный отчет

**Примечание:** Результаты являются предварительными и не заменяют консультацию врача.
""")

st.sidebar.markdown("""
### Обратная связь
Обнаружили проблему или есть предложения?  
Напишите нам: AlinaLukmanova@gmail.com
""")