import matplotlib.pyplot as plt
from io import BytesIO
import base64

def create_probability_plot(probabilities: list) -> str:
    """Создает график вероятностей и возвращает его как base64"""
    plt.figure(figsize=(8, 4))

    labels = ['Норма', 'Пневмония']
    colors = ['green', 'red']

    plt.bar(labels, probabilities, color=colors)
    plt.ylabel('Вероятность')
    plt.title('Результат классификации гибридной моделью')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)

    # Сохраняем график в base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')