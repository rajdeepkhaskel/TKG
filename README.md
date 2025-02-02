Generate a Temporal Knowledge Graph (TKG) from news articles. This is specifically for a sample from the Russia - Ukraine war. 
Use diffbot to get the csv file for the required news article.
Then, run the files in the order of the number mentioned at the end of their name.
- mistral_tkg(1).py - Generates the Timestamps alongwith the Questions and Answers. Uses mistral-large-latest model. (Generate the API key to use it)
- timestamp_formatter(2).py - Formats the timestamps and questions and answers by agai reading the article. Uses mistral-large-latest model.
- Triplex_TKG(3).ipynb - Run this in colab (if possible), and take the files from the /content/. Uses the Sciphi/Triplex model.
- mistral_tkg_refiner(4).py - Refines the final TKG by adding any missing or updating any wrong details./
All of the data is kept in the *data* folder.