# ChemAI: Predict the Cure

Хакатон ML-проект для соревнования ChemAI на Kaggle.

Задача: по числовым молекулярным дескрипторам предсказать три показателя для
каждого химического соединения:

- `IC50`
- `CC50`
- `SI`

Метрика соревнования:

```text
score = (RMSE(IC50) + RMSE(CC50) + RMSE(SI)) / 3
```

Чем меньше score, тем лучше.

## Данные

Файлы данных должны лежать в корне проекта:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

В `train.csv` целевые колонки называются:

- `IC50, mM`
- `CC50, mM`
- `SI`

Формат submission:

```text
index,IC50,CC50,SI
```

Важно: данные закрытые, поэтому `train.csv`, `test.csv` и
`sample_submission.csv` добавлены в `.gitignore`.

## Текущий лучший результат

Лучший public score на Kaggle сейчас:

```text
277.59118
```

Файл:

```text
submissions/submission_diverse_raw_group_pubshape_clip.csv
```

В названии `pubshape` означает post-processing под форму public leaderboard:
мы посмотрели на результаты public-сабмитов и поняли, что raw-модели дают
полезный сигнал, но слишком большие хвостовые предсказания ухудшают score.
Поэтому `pubshape_clip` — это не новая модель, а версия raw-предсказаний с
clipping хвостов `IC50`, `CC50` и `SI`.

Идея текущего лучшего решения:

- обучаем модели на исходных таргетах, то есть `raw`, без `log1p` и `sqrt`;
- используем ансамбль `KernelRidge + ExtraTreesRegressor`;
- усредняем предсказания моделей;
- обрезаем слишком большие значения:

```text
IC50 <= 2000
CC50 <= 3200
SI   <= 500
```

Такой clipping помогает убрать опасные выбросы в предсказаниях.

## Что делает pipeline

Основной код находится в:

- `src/chemai/pipeline.py`
- `src/train.py`

Pipeline:

1. читает `train.csv`, `test.csv`, `sample_submission.csv`;
2. удаляет константные признаки;
3. заполняет пропуски медианой;
4. добавляет индикаторы пропусков;
5. обучает выбранные модели;
6. считает локальную CV-метрику как на Kaggle;
7. усредняет fold-предсказания;
8. сохраняет submission.

Заполнение пропусков делается внутри CV-пайплайна: медиана считается только на
train-fold, затем применяется к validation-fold и test. Так validation не
используется при расчёте статистик preprocessing.

По умолчанию используется `GroupKFold` по хешу дескрипторов. Это нужно, чтобы
одинаковые наборы дескрипторов не попадали одновременно в train и validation.
Такая валидация достаточно строгая и местами пессимистичная.

## Установка

```bash
python3 -m pip install --user -r requirements.txt
```

На macOS для `LightGBM` и `XGBoost` может понадобиться OpenMP:

```bash
brew install libomp
```

Если необязательные зависимости не установлены, часть моделей будет недоступна, но
основной sklearn-код всё равно можно запускать.

## Быстрый запуск

Быстрый эксперимент:

```bash
python3 src/train.py --preset fast --models auto --cv group
```

Более полный эксперимент:

```bash
python3 src/train.py --preset full --models auto --cv group --top-k 4
```

Параметр `--top-k` означает верхнюю границу числа моделей в ансамбле. Скрипт
проверяет `top-1`, `top-2`, ..., `top-k` по out-of-fold score и выбирает лучший
вариант.

Сделать быстрый smoke run:

```bash
python3 src/train.py --preset fast --models ridge --cv group --n-splits 3
```



## Воспроизведение текущего подхода

Обучить семейство моделей, на котором основан текущий лучший подход:

```bash
python3 src/train.py \
  --preset full \
  --models kernel_ridge,extra_trees \
  --cv group \
  --top-k 2 \
  --target-transform raw \
  --submission-path submissions/submission.csv
```

Применить clipping:

```bash
python3 src/apply_clipping.py \
  --input-submission submissions/submission.csv \
  --output-submission submissions/submission_clipped.csv \
  --ic50-cap 2000 \
  --cc50-cap 3200 \
  --si-cap 500
```

После этого файл `submissions/submission_clipped.csv` можно отправлять на Kaggle.

Точная команда, которой был получен текущий лучший public submission:

```bash
python3 src/train.py \
  --preset full \
  --models catboost,lightgbm,xgboost,extra_trees,hist_gradient,gradient_boosting,svr,knn,kernel_ridge \
  --cv group \
  --n-splits 5 \
  --top-k 8 \
  --target-transform raw \
  --submission-path submissions/submission_diverse_raw_group.csv \
  --report-path reports/diverse_raw_group_summary.md \
  --output-dir outputs/diverse_raw_group
```

После обучения был применён clipping:

```bash
python3 src/apply_clipping.py \
  --input-submission submissions/submission_diverse_raw_group.csv \
  --output-submission submissions/submission_diverse_raw_group_pubshape_clip.csv \
  --ic50-cap 2000 \
  --cc50-cap 3200 \
  --si-cap 500
```

## Преобразования таргетов (Target transform)

Мы проверяли три способа обучать модели на таргетах:

```text
raw    -> модель учится на исходных значениях
sqrt   -> модель учится на sqrt(target)
log1p  -> модель учится на log(1 + target)
```

Так как Kaggle считает RMSE на исходной шкале, лучшим оказался `raw`-подход, но
с clipping хвостов.

Примеры запусков:

```bash
python3 src/train.py --preset full --models catboost,extra_trees,kernel_ridge --cv group --top-k 3 --target-transform sqrt
python3 src/train.py --preset full --models kernel_ridge,extra_trees,catboost --cv group --top-k 3 --target-transform raw
```

## Что пробовали и что не сработало

Пробовали модели:

- `CatBoost`
- `LightGBM`
- `XGBoost`
- `ExtraTreesRegressor`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`
- `GradientBoostingRegressor`
- `KernelRidge`
- `SVR`
- `KNN`
- `Ridge`
- `ElasticNet`

Лучший текущий public-результат дали `KernelRidge + ExtraTreesRegressor`.

Стратегии, которые ухудшили score:

- exact-match blending;
- refit модели на всём train;
- отдельный CatBoost под каждый таргет;
- пересчёт `SI = CC50 / IC50` вместо прямого прогноза `SI`.

Подробное описание текущего статуса:

```text
reports/current_status_team_brief.md
```

План следующих сабмитов:

```text
reports/next_submission_plan.md
```

## Тесты

Перед изменениями и после них стоит запускать:

```bash
python3 -m pytest tests
```

Тесты проверяют:

- схему данных;
- формат submission;
- реализацию Kaggle RMSE;
- преобразования `raw/sqrt/log1p`;
- связь `SI = CC50 / IC50` в train;
- лёгкий end-to-end smoke test на `Ridge`.

Проверки, которым нужны закрытые CSV-файлы, автоматически пропускаются, если
`train.csv`, `test.csv` и `sample_submission.csv` не лежат в корне проекта.
После добавления данных локально они начнут выполняться полностью.

Сейчас тестов 10, они выполняются быстро.

## Файлы и папки

```text
src/                         основной код
tests/                       тесты
reports/                     описания экспериментов и статус проекта
submissions/                 лучшие/референсные submission-файлы
requirements.txt             зависимости
README.md                    это описание
```

Папки `outputs/`, большинство файлов из `submissions/`, локальное окружение
`.venv/` и данные игнорируются через `.gitignore`.
