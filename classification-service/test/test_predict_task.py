from worker.predict_task import predict_task


def test_predict_task_format(monkeypatch):
    class DummyModel:
        def predict(self, image_bytes):
            return {
                'prediction': 1,
                'probability': 0.95,
                'probabilities': [0.05, 0.95],
                'logits': [[0.1, 2.9]]
            }

    class DummyPlot:
        @staticmethod
        def create_probability_plot(probs):
            return 'base64string'

    monkeypatch.setattr('worker.predict_task.model', DummyModel())
    monkeypatch.setattr('worker.predict_task.create_probability_plot', DummyPlot.create_probability_plot)
    monkeypatch.setattr('worker.predict_task.init_mlflow', lambda: None)
    monkeypatch.setattr('worker.predict_task.ensure_model_file_exists', lambda: None)
    result = predict_task(b'data', 'test.png')
    assert result['status'] == 'success'
    assert 'prediction' in result
    assert 'plot_image' in result
    assert 'recommendation' in result
