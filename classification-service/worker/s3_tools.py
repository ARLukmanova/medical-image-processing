import uuid

import boto3

from worker.parameters import S3_STORAGE_BUCKET, PROD_IMAGES_FOLDER
from logger import logger


def save_image_to_s3(image_bytes, filename):
    s3_filename = f"{PROD_IMAGES_FOLDER}/{uuid.uuid4().hex}_{filename}"
    logger.info(f"Сохраняем файл {filename} в S3 бакет {S3_STORAGE_BUCKET} под именем {s3_filename}")
    session = boto3.session.Session()
    s3 = session.client('s3')
    s3.put_object(
        Bucket=S3_STORAGE_BUCKET,
        Key=f"{PROD_IMAGES_FOLDER}/{uuid.uuid4().hex}_{filename}",
        Body=image_bytes,
        ContentType='image/jpeg'
    )
    logger.info(f"Файл {s3_filename} сохранен в S3 бакет {S3_STORAGE_BUCKET}")
