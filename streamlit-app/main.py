import io

import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image

from configuration.settings import settings

RESULT_POLLING_MAX_RETRIES = 30
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
    .stSpinner > div > div {
        color: #3498db !important;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown("<h1 class='title'>Анализ рентгеновских снимков на пневмонию</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        Загрузите рентгеновские снимки грудной клетки, и мы определим, есть ли признаки пневмонии, используя гибридную модель.
    </div>
""", unsafe_allow_html=True)


# Функция для отправки изображения на сервер
def predict_image(image_bytes):
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(settings.get_predict_endpoint(), files=files)

        if response.status_code == 202:
            task_id = response.json().get('task_id')
            return get_prediction_results(task_id)
        else:
            error_detail = response.json().get('detail', 'Неизвестная ошибка')
            st.error(f"Ошибка сервера для снимка: {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API-сервису. Убедитесь, что он запущен.")
        return None
    except Exception as e:
        st.error(f"Ошибка при отправке запроса: {str(e)}")
        return None


def get_prediction_results(task_id):
    import time
    for _ in range(RESULT_POLLING_MAX_RETRIES):
        poll_response = requests.get(settings.get_prediction_result_endpoint(task_id))
        if poll_response.status_code == 200:
            return poll_response.json().get('result')
        elif poll_response.status_code == 202:
            time.sleep(1)
        else:
            error_detail = poll_response.json().get('error', 'Сервис не вернул описание ошибки')
            st.error(f"Ошибка при получении результата: {error_detail}")
            return None
    st.error("Время ожидания результата истекло.")
    return None


# Функция для создания графика (показывает только Hybrid)
def create_probability_plot(pneumonia_prob, normal_prob):
    # Reduced figsize for a smaller plot
    fig, ax = plt.subplots(figsize=(4, 2)) # Changed from (8, 4) to (6, 3)

    labels = ['Норма', 'Пневмония']
    probabilities = [normal_prob, pneumonia_prob]
    colors = ['green', 'red']

    ax.bar(labels, probabilities, color=colors)
    ax.set_ylabel('Вероятность')
    ax.set_title('Результат классификации гибридной моделью')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    return fig


# --- File Uploader for Multiple Files ---
uploaded_files = st.file_uploader(
    "Выберите рентгеновские снимки (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True, # Key change here!
    help="Загрузите один или несколько рентгеновских снимков грудной клетки для анализа"
)

if uploaded_files: # Check if any files are uploaded
    # Button to trigger analysis for all uploaded files
    if st.button("Анализировать все снимки", type="primary"):
        st.subheader("Результаты анализа:")
        for i, uploaded_file in enumerate(uploaded_files):
            # Display uploaded image and prepare for analysis
            image = Image.open(uploaded_file)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            # Use an expander for each image's results
            with st.expander(f"Результат для снимка: **{uploaded_file.name}**"):
                st.image(image, caption=f"Ваш рентгеновский снимок: {uploaded_file.name}", width=300)

                with st.spinner(f"Анализируем снимок {uploaded_file.name}..."):
                    result = predict_image(img_bytes)

                    if result:
                        prediction = result.get("prediction", {})
                        recommendation = result.get("recommendation", {})
                        #plot_image_base64 = result.get("plot_image")

                        diagnosis_class_name = prediction.get("class", "Неизвестно")
                        pneumonia_probability = prediction.get("pneumonia_probability", 0.0)
                        normal_probability = prediction.get("normal_probability", 0.0)
                        is_pneumonia = prediction.get("is_pneumonia", False)

                        action_text = recommendation.get("action", "Нет рекомендаций")
                        urgency_text = recommendation.get("urgency", "Неизвестно")

                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

                        diagnosis_css_class = "pneumonia" if is_pneumonia else "normal"
                        st.markdown(f"""
                            <div class='diagnosis {diagnosis_css_class}'>
                                ЗАКЛЮЧЕНИЕ: {diagnosis_class_name.upper()}
                            </div>
                            <div class='probability'>
                                Вероятность пневмонии: <strong>{pneumonia_probability * 100:.1f}%</strong><br>
                                Вероятность нормы: <strong>{normal_probability * 100:.1f}%</strong>
                            </div>
                        """, unsafe_allow_html=True)

                        rec_css_class = "warning" if is_pneumonia else "success"
                        st.markdown(f"""
                            <div class='recommendation {rec_css_class}'>
                                <h4>РЕКОМЕНДАЦИИ</h4>
                                <p>{action_text}</p>
                                <p>Срочность: <b>{urgency_text}</b></p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                        #if plot_image_base64:
                        #    try:
                        #        plot_bytes = base64.b64decode(plot_image_base64)
                        #        st.image(plot_bytes, caption="График вероятностей от гибридной модели", use_container_width=True)
                        #    except Exception as e:
                        #        st.error(f"Ошибка при отображении графика для {uploaded_file.name}: {e}")
                        #else:
                        #    fig = create_probability_plot(pneumonia_probability, normal_probability)
                        #    if fig:
                        #        st.pyplot(fig)
                        #        plt.close(fig) # Close the figure to free up memory

                    st.markdown("---") # Separator for clarity between image results

# --- Global Interpretation and Sidebar ---
with st.expander("Как интерпретировать результаты"):
    st.markdown("""
        - **Вероятность пневмонии 0-20%** - норма
        - **20-40%** - минимальные изменения, рекомендуется контроль
        - **40-70%** - умеренные изменения, требуется обследование
        - **70-100%** - высокая вероятность патологии, срочно к врачу

        Используется гибридная модель (ResNet + EfficientNet) для более точной диагностики.
    """)

# Информация о приложении в сайдбаре
st.sidebar.markdown("""
### О приложении
Это приложение использует **гибридную нейросеть** для анализа рентгеновских снимков грудной клетки на признаки пневмонии.

**Используемая модель:**
- **Hybrid** (комбинация CNN и ViT)

**Как использовать:**
1. Загрузите **один или несколько** рентгеновских снимков.
2. Нажмите кнопку "Анализировать все снимки".
3. Получите подробный отчет для каждого снимка.

**Примечание:** Результаты являются предварительными и не заменяют консультацию врача.
""")

st.sidebar.markdown("""
### Обратная связь
Обнаружили проблему или есть предложения?  
Напишите нам: AlinaLukmanova@gmail.com
""")