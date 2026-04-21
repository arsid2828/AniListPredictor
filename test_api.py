import requests

query_staff = """
query ($search: String) {
  Staff(search: $search) {
    id
    name { full }
  }
}
"""
res2 = requests.post('https://graphql.anilist.co', json={'query': query_staff, 'variables': {'search': 'Oda Eiichiro'}})
print('Staff:', res2.json())

# Search studio
query_studio = """
query ($search: String) {
  Studio(search: $search) {
    id
    name
  }
}
"""
res3 = requests.post('https://graphql.anilist.co', json={'query': query_studio, 'variables': {'search': 'MAPPA'}})
print('Studio:', res3.json())

query_media_by_studio = """
query ($studioId: Int) {
  Page(perPage: 3) {
    media(studioId: $studioId) {
      title { romaji }
    }
  }
}
"""
if 'data' in res3.json() and res3.json()['data']['Studio']:
    s_id = res3.json()['data']['Studio']['id']
    res4 = requests.post('https://graphql.anilist.co', json={'query': query_media_by_studio, 'variables': {'studioId': s_id}})
    print('Media by Studio:', res4.json())

