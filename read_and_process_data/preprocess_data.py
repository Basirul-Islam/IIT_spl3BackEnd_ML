import nltk
#nltk.download('stopwords')
##### added section by bashir ended ####
stopwords = nltk.corpus.stopwords.words("english")

#extending the stopwords to include other words used in twitter such as retweet(rt) etc.
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
from nltk.stem.porter import *
stemmer = PorterStemmer()

def preprocess(comments):
    #print("Input Data\n", comments)
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    comment_space = comments.str.replace(regex_pat, ' ')

    # removal of @name[mention]
    regex_pat = re.compile(r'@[\w\-]+')
    comment_name = comment_space.str.replace(regex_pat, '')

    # removal of links[https://abc.com]
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    comments = comment_name.str.replace(giant_url_regex, '')

    # removal of punctuations and numbers
    punc_remove = comments.str.replace("[^a-zA-Z]", " ")
    # remove whitespace with a single space
    newtcomment=punc_remove.str.replace(r'\s+', ' ')
    # remove leading and trailing whitespace
    newtcomment=newtcomment.str.replace(r'^\s+|\s+?$','')
    # replace normal numbers with numbr
    newtcomment=newtcomment.str.replace(r'\d+(\.\d+)?','numbr')
    # removal of capitalization
    comment_lower = newtcomment.str.lower()

    # tokenizing
    tokenized_comment = comment_lower.apply(lambda x: x.split())

    # removal of stopwords
    tokenized_comment=  tokenized_comment.apply(lambda x: [item for item in x if item not in stopwords])

    # stemming of the tweets
    tokenized_comment = tokenized_comment.apply(lambda x: [stemmer.stem(i) for i in x])

    for i in range(len(tokenized_comment)):
        tokenized_comment[i] = ' '.join(tokenized_comment[i])
        comments_p= tokenized_comment

    return comments_p

#processed_comments = preprocess(comments)

#dataset['processed_comments'] = processed_comments
#print(dataset[["processed_comments"]].head(10))