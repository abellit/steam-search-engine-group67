import nltk

packages = ['stopwords', 'punkt_tab', 'wordnet']

for package in packages:
    nltk.download(package, quiet=False)

print("\nAll NLTK data downloaded successfully.")