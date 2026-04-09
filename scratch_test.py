import requests
import time

query_user = '''
query ($name: String) {
  User(name: $name) { id }
}
'''

query_act = '''
query ($userId: Int, $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    activities(userId: $userId, type: ANIME_LIST) {
      ... on ListActivity {
        id
        createdAt
        progress
        status
        media {
          episodes
          duration
        }
      }
    }
  }
}
'''

def fetch_all(username):
    r = requests.post('https://graphql.anilist.co', json={'query': query_user, 'variables': {'name': username}})
    user_id = r.json()['data']['User']['id']
    
    activities = []
    page = 1
    has_next = True
    start = time.time()
    
    while has_next:
        r2 = requests.post('https://graphql.anilist.co', json={'query': query_act, 'variables': {'userId': user_id, 'page': page}})
        data = r2.json()
        page_info = data['data']['Page']['pageInfo']
        acts = data['data']['Page']['activities']
        activities.extend(acts)
        has_next = page_info['hasNextPage']
        page += 1
        if page % 10 == 0:
            print(f"Fetched page {page-1}, total acts so far: {len(activities)}")
        if page > 50: # Limit for safety in test
            break
        time.sleep(0.5)
        
    print(f"Total time for {page-1} pages: {time.time() - start:.2f}s, Total acts: {len(activities)}")

fetch_all('arsid')
