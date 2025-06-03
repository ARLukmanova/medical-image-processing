import pytest
from unittest.mock import AsyncMock, patch
from api_adapter import APIError, send_to_api


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_send_to_api_error(mock_session):
    mock_post = AsyncMock()
    mock_post.__aenter__.return_value.status = 500
    mock_post.__aenter__.return_value.json = AsyncMock(return_value={"detail": "error"})
    mock_session.return_value.__aenter__.return_value.post = mock_post

    with pytest.raises(APIError):
        await send_to_api(b"fakeimage")
