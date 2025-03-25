import newspaper
from textblob import TextBlob
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient
import urllib.parse
# New imports for keyword analysis and visualization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter
import re

# Azure Blob Storage SAS URL (Replace with actual URL)
sas_url = ""

# ✅ Extracting account URL and SAS token correctly
parsed_url = urllib.parse.urlparse(sas_url)
account_url = f"{parsed_url.scheme}://{parsed_url.netloc}"  # Extract base URL
container_name = parsed_url.path.lstrip("/")  # Extract container name
sas_token = parsed_url.query  # Extract SAS token correctly

# Function to scrape headlines
def scrape_headlines():
    headlines = []
    sources = {
        "BBC": "https://www.bbc.com/news",
        "NYTimes": "https://www.nytimes.com"
    }

    for source, url in sources.items():
        try:
            paper = newspaper.build(url, memoize_articles=False)
            for i, article in enumerate(paper.articles[:50]): 
                try:
                    article.download()
                    article.parse()
                    headline = article.title.strip()
                    if headline:
                        headlines.append({"Source": source, "Headline": headline})
                except Exception as e:
                    print(f"Error processing article {i+1} from {source}: {e}")
        except Exception as e:
            print(f"Error scraping {source}: {e}")

    return headlines

# Function to analyze sentiment
def analyze_sentiment(headline):
    blob = TextBlob(headline)
    polarity = blob.sentiment.polarity  # type: ignore # Get the numerical polarity
    
    # Return numerical sentiment values:
    # 1 for positive, 0 for neutral, -1 for negative
    if polarity > 0.1:
        return 1
    elif polarity < -0.1:
        return -1
    else:
        return 0

# ✅ Corrected function to upload data
def upload_to_blob(df, filename):
    try:
        # ✅ Corrected: Use SAS token separately as credential
        blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)

        # ✅ Corrected: Get container client & blob client separately
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

        # Convert DataFrame to CSV
        csv_data = df.to_csv(index=False)
        csv_bytes = csv_data.encode("utf-8")

        # Upload CSV to Blob Storage
        blob_client.upload_blob(csv_bytes, overwrite=True)
        print(f"✅ Data successfully uploaded to Blob Storage as {filename}.")
    except Exception as e:
        print(f"❌ Error uploading to Blob Storage: {e}")
        raise  # Debugging

# New function to extract and plot keywords
def extract_and_plot_keywords(df):
    try:
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Combine all headlines
        all_headlines = ' '.join(df['Headline'].tolist())
        
        # Clean and tokenize
        all_headlines = re.sub(r'[^\w\s]', '', all_headlines.lower())
        tokens = word_tokenize(all_headlines)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Count keywords
        keyword_counts = Counter(keywords).most_common(15)
        
        # Plot - Bar Chart
        plt.figure(figsize=(12, 6))
        words = [word for word, count in keyword_counts]
        counts = [count for word, count in keyword_counts]
        
        plt.bar(words, counts, color='skyblue')
        plt.xlabel('Keywords')
        plt.ylabel('Frequency')
        plt.title('Most Common Keywords in Headlines')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the bar plot
        plt.savefig('headline_keywords_bar.png')
        print("✅ Keyword bar chart saved as 'headline_keywords_bar.png'")
        
        # Display the bar plot
        plt.show()
        
        # Create a scatter text plot
        plt.figure(figsize=(10, 8))
        
        # Create a colorful scatter text plot
        # Use random positions for scatter effect
        import random
        
        # Set a seed for reproducibility
        random.seed(42)
        
        # For scaling text size based on frequency
        max_count = max(counts)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', len(words))
        
        for i, (word, count) in enumerate(keyword_counts):
            # Random position
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            
            # Scale size by count
            size = 15 + (count / max_count * 30)
            
            # Get color from colormap
            color = cmap(i / len(words))
            
            plt.text(x, y, word, size=size, ha='center', va='center', 
                     color=color, weight='bold', alpha=0.8)
        
        # Remove axes for cleaner visualization
        plt.axis('off')
        plt.title('Keyword Cloud Visualization', size=16)
        
        # Save the scatter text plot
        plt.savefig('headline_keywords_scatter.png')
        print("✅ Keyword scatter plot saved as 'headline_keywords_scatter.png'")
        
        # Display the scatter plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error in keyword analysis: {e}")

# Main function
def collect_data():
    try:
        headlines = scrape_headlines()
        if not headlines:
            print("No headlines were scraped.")
            return

        df = pd.DataFrame(headlines)
        
        # Add an auto-incremented ID column
        df.insert(0, 'id', range(1, len(df) + 1))
        
        df["Sentiment"] = df["Headline"].apply(analyze_sentiment)
        print("\nCollected Data with Sentiment Analysis:")
        print(df)

        # Call the new function to plot keywords
        extract_and_plot_keywords(df)

        # ✅ Upload data using corrected function
        # upload_to_blob(df, "new_sentiment_analysis.csv")

    except Exception as e:
        print(f"❌ An error occurred during data collection: {e}")

# Run script
if __name__ == "__main__":
    collect_data()