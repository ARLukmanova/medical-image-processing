import base64
import io
from logger import logger
from typing import Dict, Any

from PIL import Image
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from api_adapter import APIError, send_to_api
from configuration.settings import settings, MAX_FILE_SIZE

# Инициализация бота и диспетчера
bot = Bot(token=settings.tg_bot_token)
dp = Dispatcher()


def validate_image(image_bytes: bytes) -> bool:
    """Проверка валидности изображения"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image: {e}")
        return False


async def send_results(message: Message, result: Dict[str, Any], image_bytes: bytes):
    """Отправка результатов пользователю с красивым оформлением"""
    try:
        # Основной результат
        prediction = result["prediction"]
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ: recommendation теперь берется напрямую из result ---
        recommendation = result["recommendation"]
        # ---------------------------------------------------------------------

        pneumonia_prob = prediction["pneumonia_probability"] * 100
        normal_prob = prediction["normal_probability"] * 100
        diagnosis = "ПНЕВМОНИЯ" if prediction["is_pneumonia"] else "НОРМА"

        # Создаем текст результата
        result_text = (
            f"🩺 <b>РЕЗУЛЬТАТ АНАЛИЗА</b>\n"
            f"<b>{diagnosis}</b>\n\n"
            f"📊 <b>ВЕРОЯТНОСТИ:</b>\n"
            f"▪️ Пневмония: <code>{pneumonia_prob:.1f}%</code>\n"
            f"▪️ Норма: <code>{normal_prob:.1f}%</code>\n\n"
            f"💡 <b>РЕКОМЕНДАЦИИ:</b>\n"
            f"{recommendation['action']}\n"  # <-- Используем новую переменную recommendation
            f"Срочность: <b>{recommendation['urgency']}</b>\n\n"  # <-- Используем новую переменную recommendation
            f"<i>Использована гибридная модель (CNN + ViT)</i>\n"
            f"<i>Это предварительный анализ, не заменяющий консультацию специалиста</i>"
        )

        # 1. Сначала отправляем уменьшенное изображение
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.thumbnail((512, 512))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            await message.answer_photo(
                photo=types.BufferedInputFile(
                    file=buffered.getvalue(),
                    filename="xray_thumbnail.jpg"
                ),
                caption="🖼 <b>Ваш рентгеновский снимок</b>",
                parse_mode=ParseMode.HTML
            )

        # 2. Затем отправляем график вероятностей (если есть)
        if "plot_image" in result and result["plot_image"]:
            try:
                plot_bytes = base64.b64decode(result["plot_image"])
                await message.answer_photo(
                    photo=types.BufferedInputFile(
                        file=plot_bytes,
                        filename="probabilities_plot.png"
                    ),
                    caption="📈 <b>График вероятностей</b>",
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                logger.error(f"Ошибка декодирования графика: {e}")

        # 3. Наконец отправляем текстовый результат
        await message.answer(
            text=result_text,
            parse_mode=ParseMode.HTML,
            reply_markup=types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [types.InlineKeyboardButton(
                        text="ℹ️ Как интерпретировать результаты",
                        callback_data="help_interpreting"
                    )]
                ]
            )
        )

    except KeyError as e:
        logger.error(
            f"Ошибка в структуре ответа API: Отсутствует ключ {e}. Получено: {result}")  # Added more context to error
        await message.answer(
            f"⚠️ Ошибка обработки результатов. Неверный формат данных от сервера. Отсутствует ключ: {e}")
    except Exception as e:
        logger.error(f"Ошибка отправки результатов: {e}", exc_info=True)
        await message.answer("⚠️ Получены результаты, но возникла ошибка при отображении")


# Обработчик для кнопки "Как интерпретировать результаты"
@dp.callback_query(F.data == "help_interpreting")
async def interpreting_help(callback: types.CallbackQuery):
    await callback.answer()
    await callback.message.answer(
        "📚 <b>Как интерпретировать результаты:</b>\n\n"
        "• <b>Вероятность пневмонии 0-20%</b> - норма\n"
        "• <b>20-40%</b> - минимальные изменения, рекомендуется контроль\n"
        "• <b>40-70%</b> - умеренные изменения, требуется обследование\n"
        "• <b>70-100%</b> - высокая вероятность патологии, срочно к врачу\n\n"
        "Гибридная модель сочетает преимущества CNN и ViT для более точной диагностики.",
        parse_mode=ParseMode.HTML
    )


async def process_image(message: Message, image_bytes: bytes):
    """Обработка изображения"""
    try:
        if len(image_bytes) > MAX_FILE_SIZE:
            await message.answer("❌ Файл слишком большой (макс. 10 МБ)")
            return

        if not validate_image(image_bytes):
            await message.answer("❌ Неверный или поврежденный формат изображения")
            return

        # Отправляем сообщение о начале обработки
        processing_msg = await message.answer("🔍 Анализирую снимок гибридной моделью...")

        result = await send_to_api(image_bytes)
        await send_results(message, result, image_bytes)

    except APIError as e:
        logger.error(f"API Error: {e}")
        await message.answer("⚠️ Ошибка сервера анализа. Попробуйте позже")
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}", exc_info=True)
        await message.answer("❌ Неизвестная ошибка при обработке")
    finally:
        # Удаляем сообщение о обработке, если оно было отправлено
        if 'processing_msg' in locals():
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
            except Exception as e:
                logger.error(f"Ошибка при удалении сообщения обработки: {e}", exc_info=True)


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    try:
        await message.answer(
            "Привет! Я бот для анализа рентгеновских снимков.\n"
            "Отправьте мне снимок грудной клетки, и я проверю наличие пневмонии.\n"
            "Отправь /help для справки.\n\n"
            "Используется гибридная модель (CNN + ViT)."
        )
    except Exception as e:
        logger.error(f"Ошибка в обработчике start: {e}")
        await message.answer("⚠️ Не удалось обработать команду /start")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    try:
        await message.answer(
            "Как использовать бота:\n"
            "1. Сделайте рентгеновский снимок грудной клетки\n"
            "2. Отправьте снимок в этот чат (как фото или файл)\n"
            "3. Получите анализ через 15-30 секунд\n\n"
            "📌 Требования к снимкам:\n"
            "- Формат: JPG, PNG\n"
            "- Макс. размер: 10 МБ\n"
            "- Четкое изображение без бликов\n\n"
            "⚠️ Бот использует гибридную модель ИИ и не заменяет консультацию врача!"
        )
    except Exception as e:
        logger.error(f"Ошибка в обработчике help: {e}")
        await message.answer("⚠️ Не удалось показать справку")


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Обработка фото из галереи"""
    processing_msg = None
    try:
        processing_msg = await message.answer("🔍 Анализирую снимок...")
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_data = await bot.download(file)
        image_bytes = file_data.read() if hasattr(file_data, 'read') else file_data
        logger.info(f"Получено фото: {len(image_bytes)} байт")
        await process_image(message, image_bytes)
    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}", exc_info=True)
        await message.answer("❌ Ошибка обработки фото")
    finally:
        if processing_msg:
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
            except Exception as e:
                logger.error(f"Ошибка при удалении сообщения обработки: {e}", exc_info=True)


@dp.message(F.document)
async def handle_document(message: Message):
    """Обработка документов-изображений"""
    processing_msg = None
    try:
        if not message.document.mime_type.startswith('image/'):
            await message.answer("Пожалуйста, отправьте изображение в формате JPEG или PNG")
            return

        processing_msg = await message.answer("🔍 Анализирую файл...")
        file = await bot.get_file(message.document.file_id)
        file_data = await bot.download(file)
        image_bytes = file_data.read() if hasattr(file_data, 'read') else file_data
        logger.info(f"Получен документ: {len(image_bytes)} байт")
        await process_image(message, image_bytes)
    except Exception as e:
        logger.error(f"Ошибка обработки документа: {e}", exc_info=True)
        await message.answer("❌ Ошибка обработки файла")
    finally:
        if processing_msg:
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=processing_msg.message_id)
            except Exception as e:
                logger.error(f"Ошибка при удалении сообщения обработки: {e}", exc_info=True)
