# coursework_2024-2025
Моя курсовая работа.

Содержание директории:

```ihep_collector.py``` - парсинг информации о всех конференциях, проходивших с 2015-01-01 по 2024-12-31 в файл conferences_data_with_year.csv

```works_info.py``` - парсинг информации о всех публикациях в выбранных конференциях в файл ```files2.csv```

```etm_inference.py``` - проводит кластеризацию документов по темам с помощью модели ETM, сохраняет темы в ```all_topics.csv```, а файлы с темами в ```files2_with_topics.csv```

```gemini.py``` - назначает названия темам из ```all_topics_csv```, записывает в ```alltopics_with_titles_gemini.csv```

```file_to_topics.py``` - заменяет номера тем на их названия в ```file2_with_topics.csv```

