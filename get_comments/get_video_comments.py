from googleapiclient.discovery import build

api_key = 'AIzaSyDyvYZq6ucVpRno2YBH89Tz7FYjAIbGY3s'


class comments:
    def __init__(self, id, comment, ):
        self.id = id
        self.comment = comment


def video_comments(video_id):
    youtube = build('youtube', 'v3',
                    developerKey=api_key)

    # retrieve youtube video results
    video_response = youtube.commentThreads().list(
        part='id,snippet,replies',
        videoId=video_id
    ).execute()

    # iterate video response
    comments_with_ids = []
    while video_response:

        # extracting required info
        # from each result object
        count = 0
        for item in video_response['items']:
            count = count + 1

            id = item['id']
            # Extracting comments
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments_with_id = comments(id, comment)
            comments_with_ids.append(comments_with_id.__dict__)
            # print(count)
            if count > 20:
                break
        break
    return comments_with_ids

def get__comments():
    from urllib.parse import urlparse
    #url_data = urlparse("https://www.youtube.com/watch?v=fLvJ8VdHLA0")
    url_data = urlparse("https://www.youtube.com/watch?v=CevxZvSJLk8")
    #url_data = urlparse(url)

    #print(url_data.query[2::])
    import urllib

    '''s = "https://www.youtube.com/watch?v=Uccvf3peELQ"
    #x = urllib.parse.quote(s, safe='')
    x = "https%3a%2f%2fwww.youtube.com%2fwatch%3fv%3dUccvf3peELQ"
    print("path: ", s)
    print("encoded path: ", x)
    print("decoded path: ", urllib.parse.unquote(x))'''
    # video = query["v"][0]
    video_id = url_data.query[2::]

    # Call function
    return video_comments(video_id)