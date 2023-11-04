import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from keys import NEWS_API_KEY

#get stock data from yahoo finance
def get_stock_pricing_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Use bert to classify sentiment from text
def analyze_sentiment_with_bert(text):
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_analysis = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    result = sentiment_analysis(text)[0]
    return result['label'], result['score']


# Example usage
if __name__ == "__main__":
    
    #define stock constants
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2022-12-31'

    #get stock data
    stock_data = get_stock_pricing_data(ticker, start_date, end_date)
    print("Stock Data:")
    print(stock_data.head(),'\n')

    #financial news text
    news_text = "Apple reported record-breaking sales this quarter."

    #analyze sentiment using BERT
    sentiment, confidence_score = analyze_sentiment_with_bert(news_text)
    print("\nSentiment Analysis of News Text:")
    print("Text:", news_text)
    #print("Sentiment:", sentiment)
    #print("Confidence Score:", confidence_score)
