import os
import time
import pandas as pd
from mistralai import Mistral

# Constants
MODEL_NAME = "mistral-large-latest"
THROTTLE_TIME = 1.5  # Prevent hitting free API limits

# Setup Mistral API
os.environ["MISTRAL_API_KEY"] = ""
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def load_csv(file_path):
    print("Loading CSV file...")
    df = pd.read_csv(file_path)
    df = df[['title', 'date_str', 'text']].dropna(subset=['text'])
    return df.iloc[0]  # Since there is a single row, return the first row as a Series

def load_questions_answers(file_path):
    print("Loading questions_answers.txt...")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_corrected_qa(article_text, article_date, existing_qa):
    prompt = f"""The following is an article along with some extracted Q&A pairs.

### Article:
{article_text}

### Existing Q&A:
{existing_qa}

### Task:
Format the Q&A pairs by inserting appropriate timestamps before each question. Use the article’s content to infer the most accurate timestamps based on context. If no exact timestamp is found, default to the article’s date: {article_date}. Do NOT use today's date.

### Output Format:
Timestamp: <correct timestamp>
Q: <question>
A: <answer>
"""
    
    time.sleep(THROTTLE_TIME)
    response = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    
    formatted_qa = response.choices[0].message.content.strip()
    print(formatted_qa)
    return formatted_qa

def save_to_txt(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)

def main(csv_file, qa_file):
    print("Starting process...")
    
    # Load data
    article = load_csv(csv_file)
    article_text = article['text']
    article_date = article['date_str']
    
    existing_qa = load_questions_answers(qa_file)
    
    # Process and format Q&A
    formatted_qa = generate_corrected_qa(article_text, article_date, existing_qa)
    
    # Save output
    output_file = "data/formatted_questions_answers.txt"
    save_to_txt(formatted_qa, output_file)
    print(f"Process completed. Data saved to {output_file}")

if __name__ == "__main__":
    csv_file = "data/diffbot-export.csv"
    qa_file = "data/questions_answers.txt"
    main(csv_file, qa_file)
