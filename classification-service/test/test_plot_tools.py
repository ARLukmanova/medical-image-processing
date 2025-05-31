from worker.plot_tools import create_probability_plot
import base64


def test_create_probability_plot():
    probs = [0.2, 0.8]
    img_b64 = create_probability_plot(probs)
    assert isinstance(img_b64, str)
    # Проверяем, что строка декодируется из base64
    base64.b64decode(img_b64)
