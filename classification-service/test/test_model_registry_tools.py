from unittest.mock import patch
from worker import model_registry_tools


@patch('worker.model_registry_tools.os.path.exists', return_value=False)
@patch('worker.model_registry_tools._download_model')
def test_ensure_model_file_exists_download(mock_download, mock_exists):
    model_registry_tools.ensure_model_file_exists()
    mock_download.assert_called_once()


@patch('worker.model_registry_tools._get_model_version', return_value='1')
@patch('worker.model_registry_tools._get_registry_latest_model_version', return_value='2')
@patch('worker.model_registry_tools.download_new_model_version')
@patch('worker.model_registry_tools.os.path.exists', return_value=True)
def test_ensure_model_file_exists_force_update(mock_exists, mock_download, mock_latest, mock_current):
    model_registry_tools.ensure_model_file_exists(force_model_update=True)
    mock_download.assert_called_once()


@patch('worker.model_registry_tools.mlflow.set_tracking_uri')
def test_init_mlflow(mock_set_uri):
    model_registry_tools.init_mlflow()
    assert mock_set_uri.called
