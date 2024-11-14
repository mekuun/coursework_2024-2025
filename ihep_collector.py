import requests
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig()

conf_notes_url = 'https://inspirehep.net/api/literature?size=1000&fields=publication_info.conference_record,publication_info.cnum&doc_type=conference%20paper&collaboration=ATLAS&page=1&q=&earliest_date=2021--2024'


def get_atlas_conferences():
    """Собирает список конференций и количество публикаций для каждой, сохраняет в CSV."""
    conf_notes = requests.get(conf_notes_url).json()
    root = conf_notes['hits']['hits']

    conferences = []
    for i in root:
        if 'metadata' in i and 'publication_info' in i['metadata']:
            pub_info = i['metadata']['publication_info'][0]
            if 'conference_record' in pub_info:
                url = pub_info['conference_record']['$ref']
                existing_conf = next((c for c in conferences if c['url'] == url), None)

                if not existing_conf:
                    conferences.append({'url': url, 'n_papers': 1})
                else:
                    existing_conf['n_papers'] += 1

    df = pd.DataFrame(conferences)
    df.to_csv('atlas_conferences.csv', index=False)
    print("Конференции сохранены в 'atlas_conferences.csv'")


def collect_conference_details():
    """Собирает детальную информацию по каждой конференции из списка и сохраняет в CSV."""
    conf_notes = requests.get(conf_notes_url).json()
    root = conf_notes['hits']['hits']

    conferences = []
    for i in root:
        if 'metadata' in i and 'publication_info' in i['metadata']:
            pub_info = i['metadata']['publication_info'][0]
            if 'conference_record' in pub_info:
                url = pub_info['conference_record']['$ref']
                existing_conf = next((c for c in conferences if c['url'] == url), None)

                if not existing_conf:
                    conferences.append({'url': url, 'n_papers': 1})
                else:
                    existing_conf['n_papers'] += 1

    for conf in conferences:
        data = requests.get(conf['url']).json()
        conf['series'] = data['metadata'].get('series', [{}])[0].get('name', '')
        conf['number'] = data['metadata'].get('series', [{}])[0].get('number', '')
        conf['title'] = data['metadata'].get('titles', [{}])[0].get('title', '')
        conf['opening_date'] = data['metadata'].get('opening_date', '')
        conf['closing_date'] = data['metadata'].get('closing_date', '')
        conf['acronym'] = data['metadata'].get('acronyms', [''])[0]

    df = pd.DataFrame(conferences)

    date_columns = ['opening_date', 'closing_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df.to_csv('ihep_conf_details.csv', index=False)
    print("Детали конференций сохранены в 'ihep_conf_details.csv'")

