import os
import json
import time
import pandas as pd
import re
from mistralai import Mistral

# File Paths (Adjust if needed)
CSV_FILE = "data/diffbot-export.csv"  # Article CSV
JSON_FILE = "data/temporal_knowledge_graph1.json"  # Original TKG
UPDATED_JSON_FILE = "data/updated_temporal_knowledge_graph.json"  # Output TKG

# Constants
MODEL_NAME = "mistral-large-latest"
THROTTLE_TIME = 1.5  # Prevent rate limits
CHUNK_SIZE = 5  # Process 5 timestamp entries at a time

# Setup Mistral API
os.environ["MISTRAL_API_KEY"] = ""
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Load the article from CSV
def load_article(csv_file):
    df = pd.read_csv(csv_file)
    return df.iloc[0, 0]  # Assuming article is in the first row and first column

# Load the existing Temporal Knowledge Graph
def load_tkg(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

# Format the prompt for a chunk
def format_prompt(article, tkg_chunk):
    return f"""Check and refine the following part of the Temporal Knowledge Graph (TKG) to ensure it correctly represents the article.

    **Article:**
    {article}

    **TKG Chunk:**
    {json.dumps(tkg_chunk, indent=4)}

    - Ensure all timestamps, entities, and relationships are correct.
    - Ensure that the timestamps are a proper date, and not some text. If no proper timestamp can be found, use the previous timestamp.
    - If key details are missing, add them.
    - Return only the corrected JSON.

    Respond in valid JSON format.
    """

# Process TKG in chunks using Mistral
def update_tkg(article, full_tkg):
    updated_tkg = []
    
    for i in range(0, len(full_tkg), CHUNK_SIZE):
        chunk = full_tkg[i:i + CHUNK_SIZE]
        prompt = format_prompt(article, chunk)

        time.sleep(THROTTLE_TIME)  # Prevent excessive API calls
        response = client.chat.complete(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])

        # Extract JSON from response
        response_text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", response_text, re.DOTALL)

        if match:
            try:
                corrected_chunk = json.loads(match.group(0))
                updated_tkg.extend(corrected_chunk)  # Merge results
            except json.JSONDecodeError:
                print(f"Warning: JSON parsing failed for chunk {i}-{i+CHUNK_SIZE}")
        else:
            print(f"Warning: No valid JSON detected for chunk {i}-{i+CHUNK_SIZE}")

    return updated_tkg if updated_tkg else full_tkg  # Return original if processing fails

# Main Execution
article_text = load_article(CSV_FILE)
original_tkg = load_tkg(JSON_FILE)
updated_tkg = update_tkg(article_text, original_tkg)

# Save the updated TKG
with open(UPDATED_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(updated_tkg, f, indent=4)

print(f"Updated Temporal Knowledge Graph saved to {UPDATED_JSON_FILE}")
