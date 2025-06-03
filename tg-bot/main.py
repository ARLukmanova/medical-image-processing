import asyncio

from bot import dp, bot
from logger import logger


async def main():
    """Запуск бота"""
    try:
        logger.info("Бот запущен и ожидает сообщений...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
    finally:
        await bot.close()

if __name__ == '__main__':
    asyncio.run(main())