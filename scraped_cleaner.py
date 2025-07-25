from bs4 import BeautifulSoup

with open("data/ashoka_web.html", "r", encoding="utf-8") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, "html.parser")

for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
    tag.decompose()

text = soup.get_text(separator="\n")

lines = [line.strip() for line in text.splitlines()]
cleaned_lines = [line for line in lines if line]

cleaned_text = "\n".join(cleaned_lines)

with open("data/ashoka_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Cleaned text saved")



# import spacy
# nlp = spacy.load("en_core_web_sm")

# # Custom stopword list (keep 'not')
# stopwords = nlp.Defaults.stop_words - {"not"}

# def preprocess_text(text):
#     doc = nlp(text)
#     tokens = []

#     for token in doc:
#         if token.text.lower() not in stopwords and not token.is_punct and not token.is_space:
#             tokens.append(token.lemma_.lower())

#     return tokens



# with open("data/ashoka_cleaned.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# processed_tokens = preprocess_text(raw_text)

# with open("data/ashoka_preprocessed.txt", "w", encoding="utf-8") as f:
#     f.write(" ".join(processed_tokens))

# print("Preprocessed tokens saved")




