from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
example = "The CFO has been fired, stock going up"

sentiment_results = nlp(example)
print(sentiment_results)

example = "The CFO has been fired, stock going down"

sentiment_results = nlp(example)
print(sentiment_results)

example = "Profit lower than last year"

sentiment_results = nlp(example)
print(sentiment_results)
