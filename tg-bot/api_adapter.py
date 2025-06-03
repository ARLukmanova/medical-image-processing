import asyncio
from typing import Dict, Any

import aiohttp

from configuration.settings import settings, RESULT_POLLING_MAX_RETRIES
from logger import logger


class APIError(Exception):
    """Кастомное исключение для ошибок API"""


async def send_to_api(image_bytes: bytes) -> Dict[str, Any]:
    """Отправка изображения в API"""
    try:
        logger.info(f"Отправка изображения размером {len(image_bytes)} байт в API")

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', image_bytes, filename='image.jpg', content_type='image/jpeg')
            async with session.post(settings.get_predict_endpoint(), data=data) as response:
                if response.status == 202:
                    resp_json = await response.json()
                    task_id = resp_json.get('task_id')
                    for _ in range(RESULT_POLLING_MAX_RETRIES):
                        await asyncio.sleep(1)
                        async with session.get(settings.get_prediction_result_endpoint(task_id)) as poll_response:
                            if poll_response.status == 200:
                                poll_json = await poll_response.json()
                                return poll_json.get('result')
                            elif poll_response.status == 202:
                                continue
                            else:
                                error_detail = (await poll_response.json()).get('error',
                                                                                'Сервис не вернул описание ошибки')
                                raise APIError(f"Ошибка при получении результата: {error_detail}")
                    raise APIError("Время ожидания результата истекло.")
                else:
                    error_detail = (await response.json()).get('detail', 'Неизвестная ошибка')
                    raise APIError(f"Ошибка сервера для снимка: {error_detail}")

    except Exception as e:
        logger.error(f"Ошибка при обращении к API: {e}")
        raise APIError(str(e))
