from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Cognitive Services credentials (replace with your own)
endpoint = "your_text_analytics_endpoint"
key = "your_text_analytics_key"

# Authenticate
ta_client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))

# Analyze sentiment with Cognitive Services
def analyze_with_cognitive(headlines):
    response = ta_client.analyze_sentiment(documents=headlines)
    for doc in response:
        print(f"Headline: {doc.id}, Sentiment: {doc.sentiment}, "
              f"Confidence: Positive={doc.confidence_scores.positive}, "
              f"Negative={doc.confidence_scores.negative}, Neutral={doc.confidence_scores.neutral}")

# Example usage
headlines = ["Great day for a breakthrough!", "Terrible incident reported today."]
analyze_with_cognitive(headlines)