import yfinance as yf

from keys import NEWS_API_KEY
from newsapi import NewsApiClient

import json

# get stock data from yahoo finance
def get_stock_pricing_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# get news data from newsapi
def get_news(query):
    news_client = NewsApiClient(api_key=NEWS_API_KEY)
    top_headlines = news_client.get_top_headlines(q=query,
                                                  language='en',
                                                  category='business',
                                                  )

    sources = news_client.get_sources()
    all_articles = news_client.get_everything(q=query,
                                              language='en',
                                              sort_by='relevancy',
                                              )

    return top_headlines, sources, all_articles


from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# Use bert to classify sentiment from text
def analyze_sentiment_with_bert(text):
    model = BertForSequenceClassification.from_pretrained(
        'nlptown/bert-base-multilingual-uncased-sentiment')
    tokenizer = BertTokenizer.from_pretrained(
        'nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_analysis = pipeline(
        'sentiment-analysis', model=model, tokenizer=tokenizer)
    result = sentiment_analysis(text)[0]
    return result['label'], result['score']


#import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia= SentimentIntensityAnalyzer()
def analyze_sentiment_with_vader(text):
    score = sia.polarity_scores(text)
    compound = score['compound']
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    return compound, neg, neu, pos
    

# Example usage
if __name__ == "__main__":

    # define stock constants
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2022-12-31'

    # get stock data
    stock_data = get_stock_pricing_data(ticker, start_date, end_date)
    print("Stock Data:")
    print(stock_data.head(), '\n')

    """ # get news article text
    top_headlines, sources, all_articles = get_news("AAPL")

    print('\n', top_headlines)
    with open("news_headlines.json", "w") as f:
        json.dump(top_headlines, f, indent=4)

    print('\n', sources)
    with open("news_sources.json", "w") as f:
        json.dump(sources, f, indent=4)

    print('\n', all_articles)
    with open("news_articles.json", "w") as f:
        json.dump(all_articles, f, indent=4) """
        
    news_article = {
        'title': "Apple stock dips after weak outlook for December quarter revenue",
        'content': "Apple reported fiscal fourth-quarter earnings on Thursday that beat analyst expectations for sales and earnings per share, but revealed that overall sales fell for the fourth quarter in a row. Every hardware business outside of the iPhone declined year over year, with big drops in the iPad and Mac segments. Apple shares fell over 3% in extended trading after executives signaled the company may not return to growth in the holiday quarter. Here’s how Apple did, versus LSEG (formerly Refinitiv) consensus expectations: EPS: $1.46 per share vs. $1.39 per share expected Revenue: $89.5 billion vs. $89.28 billion expected iPhone revenue: $43.81 billion vs. $43.81 billion expected Mac revenue: $7.61 billion vs. $8.63 billion expected iPad revenue: $6.44 billion vs. $6.07 billion expected Wearables revenue: $9.32 billion vs. $9.43 billion expected Services revenue: $22.31 billion vs $21.35 billion expected Gross margin: 45.2% vs. 44.5% expected Apple didn’t give formal guidance, but finance chief Luca Maestri said the company expected December quarter revenue to “be similar to” last year’s revenue. However, Apple said that the December quarter this year will have one fewer week. Analysts were looking for $122.98 billion in revenue for the December quarter, which would be a return to year-over-year growth of about 5% in Apple’s most important quarter. Net income was $22.96 billion, or $1.46 per share, versus $20.72 billion, or $1.29 per share, during the year-earlier period. The tech giant reported $383.29 billion in sales for the full fiscal year, down about 3% from the prior year. Quarterly revenue declined less than 1% in the September quarter. The company’s iPhone sales were in line with Wall Street expectations and increased more than 2% from last year. It was the only hardware line for Apple to show growth in the quarter, and the period only included about a week of iPhone 15 sales. Apple reports Q4 earnings, beats on sales and EPSWATCH NOW VIDEO00:48 Apple reports Q4 earnings, beats on sales and EPS Apple CEO Tim Cook told CNBC that the iPhone 15 was doing better than the iPhone 14 did during the September quarter last year. “If you look at iPhone 15 for that period of time and compare it to iPhone 14 for the same time in the year-ago quarter, iPhone 15 did better than iPhone 14,” Cook told CNBC’s Steve Kovach. He added that Apple’s more expensive Pro and Pro Max iPhones suffered supply constraints because of high demand. Apple’s Mac and iPad businesses both declined during the quarter. Maestri had warned on a call with analysts after third-quarter results that iPad and Mac sales would fall by double-digit percentages. Mac sales came in below Wall Street expectations, falling nearly 34% year over year. Apple held an unusual nighttime launch event for its new MacBook Pro laptops and iMac desktop last month. While sales of the new devices aren’t included in the quarter, Apple was signaling that new products could boost sales once again thanks to its new M3 chips. Cook told CNBC that the Mac comparison is to “an all-time record” fourth quarter, which followed a huge supply disruption and pushed what would have been third-quarter sales into the last quarter of 2022. “So, the comparison point here is very difficult,” he said. “I think the Mac is going to have a significantly better quarter in the December quarter. We’ve got the M3, we’ve got the new products, and we don’t have the compare phenomenon on a year-over-year basis,” Cook said, referring to an unusually strong market for Macs in 2022. Cook added the overall market for personal computers is “challenging.” Revenue from iPads was also down, falling 10% from the same period last year. Apple did not announce new products ahead of the holiday season this year. Apple’s services business was a bright spot. Apple recorded $22.31 billion in services revenue, outpacing analyst expectations and increasing more 16% from a year earlier. Apple’s services division includes online subscriptions like iCloud storage and Apple Music, as well as warranties from AppleCare. A big chunk of the business comes from its deal with Google for the default search engine on Apple’s browser, Safari, which has been highlighted in recent weeks as part of the Department of Justice antitrust case against Google. That payment to Apple is worth an estimated $19 billion this year. Cook said services components including App Store sales, advertising (including the Google deal), iCloud, payment services, and Apple Music did well in the quarter and hit an internal Apple record, signaling continued growth. “Every main service hit a record,” Cook said. Maestri also noted in a statement that Apple’s installed base of devices, or the number of iPhones, Macs, and iPads currently in active use, reached an all-time high during the quarter, although the company did not give an exact number. Analysts say that growth in Apple’s installed base suggests future growth in its services division. Cook said that Apple had over 1 billion paid subscriptions, which include both Apple’s own services as well as apps on the App Store that bill on a recurring basis. Apple’s wearables business unit includes headphones like the AirPods as well as Apple Watch sales. It, too, shrank year over year, dropping over 3%. Apple’s business in Greater China, its third largest market, is under the microscope as investors worry about increased competition from Huawei. Greater China sales were basically flat year over year. Apple reported $15.08 billion from the region, which includes Hong Kong and Taiwan. The company continues to have a huge amount of cash and cash-like securities on hand, even as it strives to offset its cash pile with debt. Apple said it had $162.1 billion in cash on hand on Thursday. Apple said it would pay a dividend of 24 cents per share this month and said the company had spent $25 billion during the quarter on share repurchases and dividends.",
        'url': "https://www.cnbc.com/2023/11/02/apple-aapl-earnings-report-q4-2023.html"
    }
    
    #print article
    print("News Article:")
    print(news_article['title'], '\n')
    print(news_article['content'], '\n')
        
    #analyze news article using BERT
    sentiment, score = analyze_sentiment_with_bert(news_article['title'])
    print("BERT Title Sentiment Analysis:")
    print(sentiment, score, '\n')
    
    #analyze news article using VADER
    compound, neg, neu, pos = analyze_sentiment_with_vader(news_article['content'])
    print("VADER Content Sentiment Analysis:")
    print(compound, neg, neu, pos, '\n')