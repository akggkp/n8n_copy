import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import os
from datetime import datetime

# Adjust the path to import from the application correctly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.tasks import validate_video
from config import Config
from app.models import ProcessedVideo

# Mock Config.DATABASE_URL to prevent actual connection attempts during testing
@pytest.fixture(autouse=True)
def mock_config_db_url():
    with patch('app.tasks.Config.DATABASE_URL', 'sqlite:///:memory:'):
        yield

def test_validate_video_success(mocker):
    """
    Test successful video validation and database registration.
    """
    video_id = "test_video_123"
    file_path = "/data/videos/test_video_123.mp4"
    filename = "test_video_123.mp4"

    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.path.getsize', return_value=1024 * 1024 * 50)

    with patch('app.tasks.db_session') as mock_db_session:
        mock_session = mock_db_session().__enter__()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        result = validate_video(video_id, file_path, filename)

        assert result['status'] == 'success'
        assert result['video_id'] == video_id
        
        mock_session.query.assert_called_once_with(ProcessedVideo)
        mock_session.query.return_value.filter_by.assert_called_once_with(video_id=video_id)
        mock_session.add.assert_called_once()
        
        added_object = mock_session.add.call_args[0][0]
        assert isinstance(added_object, ProcessedVideo)
        assert added_object.video_id == video_id
        assert added_object.filename == filename
        assert added_object.status == 'validating'


def test_validate_video_file_not_found(mocker):
    """
    Test video validation when the file does not exist.
    """
    video_id = "non_existent_video"
    file_path = "/data/videos/non_existent_video.mp4"
    filename = "non_existent_video.mp4"

    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('app.tasks.validate_video.retry', side_effect=Exception("Task retry triggered"))

    with pytest.raises(Exception, match="Task retry triggered"):
        validate_video(video_id, file_path, filename)

def test_validate_video_already_processed(mocker):
    """
    Test video validation when the video has already been processed.
    """
    video_id = "existing_video"
    file_path = "/data/videos/existing_video.mp4"
    filename = "existing_video.mp4"

    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.path.getsize', return_value=100)

    with patch('app.tasks.db_session') as mock_db_session:
        mock_session = mock_db_session().__enter__()
        mock_session.query.return_value.filter_by.return_value.first.return_value = ProcessedVideo(video_id=video_id)

        result = validate_video(video_id, file_path, filename)

        assert result['status'] == 'skipped'
        assert result['reason'] == 'Already processed'

        mock_session.add.assert_not_called()
