import os
import time
import pandas as pd
import re
from datetime import datetime
from mistralai import Mistral

# Constants
MODEL_NAME = "mistral-large-latest"
MAX_TOKENS = 8000  # Mistral can handle larger chunks
CHUNK_OVERLAP = 100  # Adjust overlap for context retention
THROTTLE_TIME = 1.5  # Prevent hitting free API limits

# Setup Mistral API
os.environ["MISTRAL_API_KEY"] = ""
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def load_csv(file_path):
    print("Loading CSV file...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Preprocessing data...")
    df = df[['title', 'date_str', 'text']]
    df['date_str'] = pd.to_datetime(df['date_str'], errors='coerce')
    return df.dropna(subset=['text'])

def extract_timestamps(text, default_date):
    date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
    dates = re.findall(date_pattern, text)
    
    if dates:
        parsed_dates = [datetime.strptime(d, "%Y-%m-%d") if "-" in d else datetime.strptime(d, "%d/%m/%Y") for d in dates]
        return min(parsed_dates).strftime("%Y-%m-%d %H:%M:%S")
    return default_date.strftime("%Y-%m-%d %H:%M:%S")

def chunk_text(text, max_tokens=MAX_TOKENS, overlap=CHUNK_OVERLAP):
    sentences = text.split(". ")
    chunks = []
    chunk = ""
    
    for i, sentence in enumerate(sentences):
        if len(chunk.split()) + len(sentence.split()) > max_tokens:
            chunks.append(chunk.strip())
            chunk = " ".join(sentences[max(0, i-overlap):i])  # Overlap previous part
        chunk += sentence + ". "
    
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def generate_qa(text, timestamp):
    prompt = f"""Generate factual Q&A pairs based on the text below. Ensure questions are context-aware.
    Generate as many Q&A pairs, such that they cover the entire text. Keep the answers concise and factual.
    Infer the correct timestamp from the text itself. If the timestamp is unclear, default to the article's date.
    
    Format:
    Timestamp: <inferred timestamp or article date>
    Q: <question>
    A: <answer>
    
    Text: {text}"""
    
    time.sleep(THROTTLE_TIME)
    response = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    
    message_content = response.choices[0].message.content.strip()
    
    # Split the response into lines
    lines = message_content.split("\n")
    
    # Check if the first two lines both start with a timestamp
    if len(lines) > 1 and lines[0].startswith("Timestamp") and lines[1].startswith("Timestamp"):
        # Remove the first line and keep the rest
        message_content = "\n".join(lines[1:])
    
    print(message_content)
    return message_content

def save_to_txt(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(f"Timestamp: {item['timestamp']}\n{item['qa']}\n\n")

def main(csv_file):
    print("Starting process...")
    df = load_csv(csv_file)
    df = preprocess_data(df)
    
    all_qa = []
    for _, row in df.iterrows():
        text = row['text']
        default_timestamp = row['date_str'] if not pd.isna(row['date_str']) else datetime.now()
        chunks = chunk_text(text)
        i, chunk_len = 0, len(chunks)
        for chunk in chunks:
            i += 1
            print(f"Processing chunk {i}/{chunk_len}...")
            qa_pairs = generate_qa(chunk, default_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            
            if qa_pairs:
                all_qa.append({"timestamp": default_timestamp.strftime("%Y-%m-%d %H:%M:%S"), "qa": qa_pairs})
    
    save_to_txt(all_qa, "data/questions_answers.txt")
    print("Process completed. Data saved to data/questions_answers.txt")

if __name__ == "__main__":
    csv_file = "data/diffbot-export.csv"
    main(csv_file)
