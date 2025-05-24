#!/bin/bash
set -e

REPO_URL="https://github.com/ARLukmanova/medical-image-processing.git"
REPO_NAME=$(basename "$REPO_URL" .git)
TARGET_DIR="$HOME/$REPO_NAME"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "Репозиторий уже существует. Обновляю..."
    cd "$TARGET_DIR"
    git pull
else
    echo "Клонирую репозиторий..."
    git clone "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
fi

echo "Выполняю dvc pull..."
dvc pull