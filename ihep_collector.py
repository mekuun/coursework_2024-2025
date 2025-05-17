import requests
import csv

conferences_url = 'https://inspirehep.net/api/conferences'
query = 'opening_date:[2015-01-01 TO 2024-12-31]'
params = {
    'q': query,
    'size': 1000,
}

response = requests.get(conferences_url, params=params)
data = response.json()

conference_list = []
for conference in data['hits']['hits']:
    conference_metadata = conference['metadata']
    conference_id = conference_metadata['control_number']
    conference_name = conference_metadata.get('titles', [{}])[0].get('title', 'No Title')
    conference_cnum = conference_metadata['cnum']
    number_of_contributions = conference_metadata.get('number_of_contributions', 0)

    opening_date = conference_metadata.get('opening_date', 'Unknown')
    year = opening_date.split('-')[0] if opening_date != 'Unknown' else 'Unknown'

    conference_list.append({
        'id': conference_id,
        'name': conference_name,
        'year': year,
        'number_of_contributions': number_of_contributions,
        'cnum': conference_cnum
    })

sorted_conferences = sorted(conference_list, key=lambda x: x['number_of_contributions'], reverse=True)

csv_filepath = "conferences_data_with_year.csv"
with open(csv_filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["id", "name", "year", "number_of_contributions", 'cnum'])
    writer.writeheader()
    writer.writerows(sorted_conferences)
