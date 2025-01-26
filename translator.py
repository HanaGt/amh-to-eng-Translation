import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from tqdm import tqdm
import requests

# URLs for downloading the dataset
urls = [
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_E-Bible/amharic.txt", "amharic_bible.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_E-Bible/english.txt", "english_bible.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_News/amharic.txt", "amharic_news.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_News/english.txt", "english_news.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_History/amharic.txt", "amharic_history.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_History/english.txt", "english_history.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_Legal/amharic.txt", "amharic_legal.txt"),
    ("https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_Legal/english.txt", "english_legal.txt")
]

# Download dataset if not already present
def download_dataset():
    for url, filename in urls:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            print(f"{filename} already exists. Skipping download.")

# Load Amharic and English texts
def load_texts(amharic_files, english_files):
    amharic_data = []
    english_data = []

    for am_file, en_file in zip(amharic_files, english_files):
        with open(am_file, "r", encoding="utf-8") as am_f, open(en_file, "r", encoding="utf-8") as en_f:
            amharic_data.extend(am_f.readlines())
            english_data.extend(en_f.readlines())

    return amharic_data, english_data

# Tokenize data
def tokenize_data(amharic_sentences, english_sentences, tokenizer):
    amharic_encodings = tokenizer(amharic_sentences, padding=True, truncation=True, return_tensors="pt")
    english_encodings = tokenizer(english_sentences, padding=True, truncation=True, return_tensors="pt")
    return amharic_encodings, english_encodings

# Custom Dataset class
class TranslationDataset(Dataset):
    def __init__(self, amharic_file, english_file, tokenizer, max_length=128):
        self.amharic_lines = open(amharic_file, 'r', encoding='utf-8').readlines()
        self.english_lines = open(english_file, 'r', encoding='utf-8').readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.amharic_lines)

    def __getitem__(self, idx):
        amharic_text = self.amharic_lines[idx].strip()
        english_text = self.english_lines[idx].strip()

        amharic_tokens = self.tokenizer(amharic_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        english_tokens = self.tokenizer(english_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        input_ids = amharic_tokens['input_ids'].squeeze()
        attention_mask = amharic_tokens['attention_mask'].squeeze()
        labels = english_tokens['input_ids'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Training function
def train(model, dataloader, optimizer, device, epochs=3):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")

# Translation function
def translate(model, tokenizer, text, device):
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=128)

    translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translated_text

# Main execution
if __name__ == '__main__':
    # Download dataset
    download_dataset()

    # List of Amharic and English files
    amharic_files = ["amharic_news.txt", "amharic_history.txt", "amharic_legal.txt"]
    english_files = ["english_news.txt", "english_history.txt", "english_legal.txt"]

    # Load and combine datasets
    amharic_sentences, english_sentences = load_texts(amharic_files, english_files)

    # Print sample data
    print("Amharic:", amharic_sentences[0].strip())
    print("English:", english_sentences[0].strip())

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")

    # Set up dataset and dataloader
    dataset = TranslationDataset('amharic_news.txt', 'english_news.txt', tokenizer)
    batch_size = 8
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train the model
    train(model, train_dataloader, optimizer, device)

    # Save the trained model
    model.save_pretrained('trained_amharic_english_model')
    tokenizer.save_pretrained('trained_amharic_english_model')

    # Evaluate the model
    evaluate(model, train_dataloader, device)

    # Example translation
    test_sentence = "እንዴት ነህ?"
    translated = translate(model, tokenizer, test_sentence, device)
    print("Original:", test_sentence)
    print("Translated:", translated)