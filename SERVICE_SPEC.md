# Спека: сервис рекомендаций игр (FastAPI + Qdrant + Docker Compose)

## Что нужно сделать

1. FastAPI-сервис с эндпоинтом `POST /recommendations`.
2. Сервис `api` в `docker-compose.yml` рядом с уже готовым `qdrant`: `docker compose up` поднимает оба.
3. По желанию: кэш ответов в Redis.

Модель и вся логика рекомендаций уже готовы и протестированы, ML писать не нужно. Сервис только вызывает `Recommender`.

## Что уже есть

| Что | Где |
|---|---|
| Логика рекомендаций | `src/serving.py`, класс `Recommender` |
| Файлы модели | `models/`: `items.parquet`, `user_side.npz`, `ranker.txt`, `manifest.json`. В git их нет, возьмите у нас |
| Qdrant | сервис `qdrant` в `docker-compose.yml` |
| Загрузка игр в Qdrant | `src/load_qdrant.py` |
| Тесты логики | `tests/test_serving.py` |

## Подготовка: один раз

```bash
docker compose up -d qdrant
uv run python -m src.load_qdrant          # ~10 с, "games: 47539 points"
```

Можно запустить без установки всего проекта (там PyTorch на ~3 ГБ):

```bash
uv run --no-project --with qdrant-client==1.20.0 --with pandas --with pyarrow python -m src.load_qdrant
```

Данные Qdrant хранятся в docker volume и переживают перезапуск. Загружать заново нужно только при новых файлах модели.

## Как вызывать логику

```python
from src.serving import Recommender

recommender = Recommender(models_dir="models", qdrant_url="http://localhost:6333")   # ОДИН раз при старте
items = recommender.recommend(history, top_k=10)
```

- `history` — список словарей `{"app_id": int, "playtime": float, "is_recommended": bool}`.
- Возвращается список словарей `{"app_id", "title", "score", "price", "tags"}`, уже в порядке рекомендации.

`Recommender` грузит модели и подключается к Qdrant. Создавайте его **один раз** при старте приложения (lifespan), а не на каждый запрос.

## API

```json
POST /recommendations
{
  "history": [
    {"app_id": 730, "playtime": 812.5, "is_recommended": true},
    {"app_id": 271590, "playtime": 50.0, "is_recommended": false}
  ],
  "top_k": 10
}
```

Правила валидации (Pydantic, при ошибке 422):

| Поле | Правило |
|---|---|
| `app_id` | `int ≥ 1` |
| `playtime` | `float ≥ 0`, **часы** |
| `is_recommended` | `bool` |
| `top_k` | `1 ≤ top_k ≤ 100`, по умолчанию 10 |
| `history` | от старых к новым, может быть пустой |

```json
{
  "items": [
    {"app_id": 1794680, "title": "Vampire Survivors", "score": 0.838, "price": 5.0, "tags": []},
    {"app_id": 1145360, "title": "Hades", "score": 0.539, "price": 25.0, "tags": []}
  ],
  "total": 10,
  "cached": false,
  "inference_time_ms": 47.0
}
```

- `items` — ответ `recommend` как есть;
- `total` — `len(items)`;
- `cached` — `true`, если ответ взят из Redis;
- `inference_time_ms` — время вызова `recommend`.

`score = null` означает fallback на популярные игры: так бывает, если ни одной игры из истории нет в каталоге. `tags` иногда пустой — это особенность датасета.

## Docker

Добавьте в `docker-compose.yml`:

```yaml
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      QDRANT_URL: http://qdrant:6333        # внутри compose — имя сервиса, не localhost
      MODELS_DIR: /app/models
    volumes:
      - ./models:/app/models:ro
    depends_on:
      - qdrant
```

Dockerfile:
- `python:3.12-slim`;
- `apt-get install -y libgomp1`, иначе LightGBM не импортируется;
- `pip install fastapi uvicorn numpy pandas lightgbm==4.7.0 qdrant-client==1.20.0`. **PyTorch не нужен**;
- `COPY src/__init__.py src/serving.py /app/src/` и ваш код приложения;
- `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.

Настройки читайте из окружения: `QDRANT_URL`, `QDRANT_API_KEY` (локально пустой), `QDRANT_COLLECTION` (`games`), `MODELS_DIR`.

## Проверка

```bash
uv run pytest tests/test_serving.py      # логика + Qdrant: должно быть 3 passed, а не skipped
python -c "import json; print(json.dumps(json.load(open('models/manifest.json'))['control_example']['request']))" \
  | curl -s -X POST localhost:8000/recommendations -H 'Content-Type: application/json' -d @-
```

`app_id` в ответе должны совпасть с `control_example.expected` из `models/manifest.json` — и по составу, и по порядку. Первые три: Vampire Survivors, Hades, CULTIC.
