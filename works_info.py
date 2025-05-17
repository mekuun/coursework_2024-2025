import requests
import csv
import pandas as pd


def fetch_publications_by_cnum(cnum, size=1000):
    base_url = 'https://inspirehep.net/api/literature'
    query = f'publication_info.cnum:{cnum}'
    params = {
        'q': query,
        'size': size,
        'sort': 'mostrecent',
    }
    response = requests.get(base_url, params=params)

    data = response.json()
    publications = []
    for article in data.get('hits', {}).get('hits', []):
        metadata = article.get('metadata', {})
        title = metadata.get('titles', [{}])[0].get('title', 'No Title')
        year = metadata.get('imprints', [{}])[0].get('date', 'Unknown').split('-')[0]
        keywords = [kw['value'] for kw in metadata.get('keywords', [])]
        abstracts = [ab['value'] for ab in metadata.get('abstracts', [])]

        publications.append({
            'title': title,
            'year': year,
            'keywords': "; ".join(keywords),
            'abstracts': "; ".join(abstracts),
        })

    return publications


def process_conference_data(csv_filepath):
    df = pd.read_csv(csv_filepath)

    for cnum in df['cnum'].unique():
        publications = fetch_publications_by_cnum(cnum)

        total_articles = len(publications)
        #articles_with_keywords = sum(1 for p in publications if p['keywords'])
        #articles_with_abstracts = sum(1 for p in publications if p['abstracts'])

        #keywords_percentage = (articles_with_keywords / total_articles * 100) if total_articles else 0
        #abstracts_percentage = (articles_with_abstracts / total_articles * 100) if total_articles else 0

        output_filepath = f"files2.csv"
        if publications:
            with open(output_filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=["title", "year", "keywords", "abstracts"])
                writer.writerows(publications)
            print(f"Публикации  успешно сохранены в {output_filepath}")
        else:
            print(f"Для конференции не найдено публикаций.")

        print(f"Конференция {cnum}:")
        print(f"Всего статей: {total_articles}")
        #print(f"Процент статей с ключевыми словами: {keywords_percentage:.2f}%")
        #print(f"Процент статей с аннотациями: {abstracts_percentage:.2f}%")
        #print("-" * 50)

process_conference_data('conferences_data_with_year.csv')