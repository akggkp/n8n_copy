import io
from config import Config
from app.web.app import app as flask_app
import pytest
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))


@pytest.fixture
def app():
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_upload_file(client, mocker):
    """Test file upload endpoint"""
    # Mock the celery task
    mock_send_task = mocker.patch('app.celery_app.celery_app.send_task')

    # Mock os.path.join to predict the file path
    mocker.patch(
        'os.path.join',
        return_value=os.path.join(
            Config.VIDEO_WATCH_DIR,
            'test.mp4'))

    # Mock file.save
    mock_save = mocker.patch('werkzeug.datastructures.FileStorage.save')

    data = {
        'file': (io.BytesIO(b"some initial text data"), 'test.mp4')
    }

    response = client.post(
        '/api/upload',
        data=data,
        content_type='multipart/form-data')

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'success'
    assert json_data['filename'] == 'test.mp4'

    # Check that the task was called
    mock_send_task.assert_called_once()

    # Check that file.save was called
    mock_save.assert_called_once_with(
        os.path.join(
            Config.VIDEO_WATCH_DIR,
            'test.mp4'))
