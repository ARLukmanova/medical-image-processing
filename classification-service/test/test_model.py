import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from worker.model import Model


@pytest.fixture
def fake_image_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (224, 224)).save(buf, format='PNG')
    return buf.getvalue()


@patch('worker.model.ort.InferenceSession')
@patch('os.path.exists', return_value=True)
def test_model_init(mock_exists, mock_ort):
    mock_ort.return_value.get_inputs.return_value = [MagicMock(name='input', shape=[1, 3, 224, 224])]
    model = Model()
    assert hasattr(model, 'ort_session')
    assert hasattr(model, 'input_name')
    assert hasattr(model, 'model_input_size')


@patch('worker.model.ort.InferenceSession')
@patch('os.path.exists', return_value=True)
def test_predict_success(mock_exists, mock_ort, fake_image_bytes):
    mock_input = MagicMock()
    mock_input.shape = [1, 3, 224, 224]
    mock_input.name = 'input'
    mock_ort.return_value.get_inputs.return_value = [mock_input]
    mock_ort.return_value.run.return_value = [np.array([[0.1, 0.9]])]
    model = Model()
    result = model.predict(fake_image_bytes)
    assert 'prediction' in result
    assert 'probability' in result
    assert 'probabilities' in result
    assert 'logits' in result


def test_check_content_length_raises():
    from worker.model import Model
    m = object.__new__(Model)
    with pytest.raises(Exception):
        m._check_content_length(b'0' * (11 * 1024 * 1024))
