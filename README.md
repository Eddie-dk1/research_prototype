# Исследовательский прототип НИР

## 1. Тема НИР

Разработка нейросетевой системы для прогнозирования кибератак и выявления аномального поведения пользователей на основе анализа сетевых логов и пользовательской активности.

## 2. Цель промежуточного прототипа

Цель данного промежуточного прототипа состоит в демонстрации первого рабочего исследовательского pipeline для задачи обнаружения аномалий. Прототип охватывает генерацию синтетических логов, предобработку данных, формирование признаков, обучение baseline-моделей, обучение LSTM-Autoencoder, расчет метрик качества и сохранение итоговых артефактов.

## 3. Структура проекта

```text
research_prototype/
├── docs/
│   ├── project_context.md
│   └── PDF-материалы по НИР
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── evaluate.py
│   ├── explain_anomalies.py
│   ├── features.py
│   ├── generate_data.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── train_lstm_autoencoder.py
│   └── utils.py
├── results/
│   └── plots/
├── requirements.txt
└── README.md
```

## 4. Команды запуска

### Установка зависимостей

```bash
python3 -m pip install -r requirements.txt
```

### Быстрый запуск одним скриптом

В корне проекта есть скрипт `run_all.sh`, который запускает весь pipeline одной командой:

```bash
bash run_all.sh
```

Полезные варианты:

```bash
bash run_all.sh --best-autoencoder
bash run_all.sh --skip-generate --best-autoencoder
bash run_all.sh --with-stage10
bash run_all.sh --skip-generate --with-stage10 --stage10-epochs 50
```

Ключи:

- `--skip-generate` пропускает генерацию синтетических логов, если `data/raw/synthetic_logs.csv` уже существует;
- `--best-autoencoder` обучает лучший практический вариант LSTM-Autoencoder вместо дефолтной конфигурации;
- `--with-stage10` запускает всю последовательность экспериментов этапа 10;
- `--stage10-epochs 50` позволяет задать число эпох для эксперимента 6 в stage 10.

### Последовательный запуск этапов вручную

```bash
python3 src/generate_data.py
python3 src/preprocessing.py
python3 src/train_baseline.py
python3 src/train_lstm_autoencoder.py
python3 src/evaluate.py
python3 src/explain_anomalies.py
```

Если в системе настроен алиас `python`, допускается запуск тех же команд через `python`, однако для macOS в рамках данного прототипа рекомендуется использовать `python3`.

### Лучшая практическая конфигурация автоэнкодера

Если нужен не дефолтный, а лучший найденный практический вариант автоэнкодера, используйте:

```bash
python3 src/train_lstm_autoencoder.py \
  --sequence-length 5 \
  --hidden-size 32 \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --threshold-percentiles 85 90 92 95 \
  --early-stopping-patience 0 \
  --selection-metric f1
python3 src/evaluate.py
```

## 5. Какие данные используются

В проекте используется синтетический датасет сетевых и пользовательских событий, формируемый скриптом `src/generate_data.py`. Датасет содержит нормальные события и несколько типов аномалий:

- `brute_force_login`
- `unusual_night_activity`
- `data_exfiltration`
- `unusual_ip_change`

Сырые данные сохраняются в файл `data/raw/synthetic_logs.csv`.

## 6. Какие модели реализованы

В прототипе реализованы три модели:

- `Isolation Forest` как unsupervised baseline;
- `Random Forest` как supervised baseline;
- `LSTM-Autoencoder` на PyTorch для реконструкции последовательностей событий.

Отдельный модуль `src/explain_anomalies.py` формирует таблицу примеров аномалий с краткими интерпретируемыми причинами.

## 7. Какие метрики считаются

Для оценки качества моделей рассчитываются следующие метрики:

- `precision`
- `recall`
- `f1`
- `roc_auc`

Дополнительно сохраняются матрицы ошибок и сравнительные графики по качеству моделей.

## 8. Какие файлы появляются в results

После полного запуска pipeline формируются следующие основные результаты:

- `results/metrics_baseline.json`
- `results/metrics_autoencoder.json`
- `results/metrics_summary.csv`
- `results/anomaly_examples.csv`
- `results/baseline_outputs.csv`
- `results/autoencoder_outputs.csv`
- `results/autoencoder_history.csv`
- `results/autoencoder_metadata.json`
- `results/lstm_autoencoder.pt`

В каталоге `results/plots/` сохраняются графики:

- `confusion_matrix_random_forest.png`
- `confusion_matrix_isolation_forest.png`
- `confusion_matrix_autoencoder.png`
- `roc_curve_baseline.png`
- `roc_curve_autoencoder.png`
- `reconstruction_error_distribution.png`
- `f1_score_comparison.png`
- `roc_auc_comparison.png`

Для подготовки раздела «Промежуточные результаты исследования» в отчете в первую очередь рекомендуется использовать:

- `results/metrics_summary.csv`
- `results/metrics_baseline.json`
- `results/metrics_autoencoder.json`
- `results/anomaly_examples.csv`
- графики из `results/plots/`

## 9. Ограничения прототипа

- используются синтетические данные, а не реальные журналы событий;
- результаты являются предварительными и предназначены для промежуточного этапа НИР;
- прототип не является промышленной IDS/UEBA-системой;
- качество LSTM-Autoencoder на текущем этапе заметно уступает baseline-моделям;
- необходима последующая проверка на открытых датасетах, например CICIDS2017, CSE-CIC-IDS2018 или CERT Insider Threat Dataset.

## 10. Что можно улучшить на следующем этапе

- использовать более реалистичные и разнообразные сценарии аномалий;
- заменить простое integer-кодирование категориальных признаков на более информативные представления;
- доработать формирование последовательностей и подбор длины окна;
- улучшить архитектуру LSTM-Autoencoder и процедуру выбора threshold;
- провести тестирование на открытых датасетах;
- добавить сравнение с дополнительными моделями и более подробный анализ ошибок.

## Используемые технологии

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- PyTorch
- tqdm
