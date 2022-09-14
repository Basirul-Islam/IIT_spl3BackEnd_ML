from googleapiclient.discovery import build

api_key = 'AIzaSyDyvYZq6ucVpRno2YBH89Tz7FYjAIbGY3s'


class comments:
    def __init__(self, id, comment, ):
        self.id = id
        self.comment = comment


def video_comments(video_id):
    # empty list for storing reply
    # replies = []

    # creating youtube resource object
    youtube = build('youtube', 'v3',
                    developerKey=api_key)

    # retrieve youtube video results
    video_response = youtube.commentThreads().list(
        part='id,snippet,replies',
        videoId=video_id
    ).execute()

    # iterate video response
    #ids = []
    #comments = []
    comments_with_ids = []
    while video_response:

        # extracting required info
        # from each result object
        count = 0
        for item in video_response['items']:
            count = count + 1

            id = item['id']
            #ids.append(id)
            # print(id)
            # Extracting comments
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            #comments.append(comment)
            comments_with_id = comments(id, comment)
            #print(comments_with_id.__dict__)
            #print(id,comment)
            comments_with_ids.append(comments_with_id.__dict__)
            # print(count)
            if count > 20:
                break

            # counting number of reply of comment
            # replycount = item['snippet']['totalReplyCount']

            # if reply is there
            '''if replycount > 0:
                # iterate through all reply
                for reply in item['replies']['comments']:
                    # Extract reply
                    reply = reply['snippet']['textDisplay']'''
        # print(comments)
        break

        # Store reply is list
        # replies.append(reply)

        # print comment with list of reply
        # print(comment, end='\n\n')
        # rows = [id, comment]
        # csvwriter.writerows(rows)
        # empty reply list
        # replies = []
    '''import pandas as pd
    import numpy as np
    a = np.array(ids)
    b = np.array(comments)

    df = pd.DataFrame({"id": a, "comment": b})
    df.to_csv("comments.csv", index=False)

    import pandas as panda
    datasetToPredict = panda.read_csv("comments.csv")
    x = datasetToPredict.head(2)
    print(comments_with_ids)'''
    return comments_with_ids
    # Again repeat
    '''if 'nextPageToken' in video_response:
               video_response = youtube.commentThreads().list(
                   part='snippet,replies',
                   videoId=video_id
               ).execute()
           else:
               break'''


# Enter video id

def get__comments():
    from urllib.parse import urlparse
    url_data = urlparse("https://www.youtube.com/watch?v=koxSCBheIjc")
    #url_data = urlparse(url)

    #print(url_data.query[2::])
    import urllib

    s = "https://www.youtube.com/watch?v=Uccvf3peELQ"
    #x = urllib.parse.quote(s, safe='')
    x = "https%3a%2f%2fwww.youtube.com%2fwatch%3fv%3dUccvf3peELQ"
    print("path: ", s)
    print("encoded path: ", x)
    print("decoded path: ", urllib.parse.unquote(x))
    # video = query["v"][0]
    video_id = url_data.query[2::]

    # Call function
    return video_comments(video_id)
