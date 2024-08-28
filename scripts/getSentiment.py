from textblob import TextBlob
def get_sentiment(text):
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text}")
        print(e)
        return None

def classify_sentiment_score(score):
    if score is not None:
        if score>0:
            return 'Positive'
        elif score<0:
            return 'Negative'
        else:
            return 'Neutral'
    else:
        return 'N/A'