import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import MarianMTModel, MarianTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from sacrebleu import corpus_bleu

# Correct Model Selection
MODEL_NAME = "Helsinki-NLP/opus-mt-amh-en"  # Amharic → English

# Load tokenizer & model
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset
def load_texts(am_file, en_file):
    with open(am_file, "r", encoding="utf-8") as am_f, open(en_file, "r", encoding="utf-8") as en_f:
        amharic_data = am_f.readlines()
        english_data = en_f.readlines()
    return amharic_data, english_data

# Files
amharic_files = ["amharic_bible.txt", "amharic_history.txt", "amharic_legal.txt"]
english_files = ["english_bible.txt", "english_history.txt", "english_legal.txt"]

# Combine all files
amharic_sentences, english_sentences = [], []
for am_file, en_file in zip(amharic_files, english_files):
    am, en = load_texts(am_file, en_file)
    amharic_sentences.extend(am)
    english_sentences.extend(en)

# Ensure equal dataset size
assert len(amharic_sentences) == len(english_sentences), "Mismatch in dataset sizes!"

# Define Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, amharic_texts, english_texts, tokenizer, max_length=128):
        self.amharic_texts = amharic_texts
        self.english_texts = english_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.amharic_texts)

    def __getitem__(self, idx):
        am_text = self.amharic_texts[idx].strip()
        en_text = self.english_texts[idx].strip()
        
        # Tokenize
        inputs = self.tokenizer(am_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(en_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

# Split dataset into train (80%) and validation (20%)
total_size = len(amharic_sentences)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

dataset = TranslationDataset(amharic_sentences, english_sentences, tokenizer)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Function
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
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

# Evaluation with BLEU Score
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_loss = 0
    references, hypotheses = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            # Generate translations
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            translated_texts = [tokenizer.decode(gid, skip_special_tokens=True) for gid in generated_ids]
            actual_texts = [tokenizer.decode(lid, skip_special_tokens=True) for lid in labels]
            hypotheses.extend(translated_texts)
            references.extend([[ref] for ref in actual_texts])  # BLEU expects list of references
    avg_loss = total_loss / len(dataloader)
    bleu_score = corpus_bleu(hypotheses, references).score
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"BLEU Score: {bleu_score:.2f}")

# Translation Function
def translate(model, tokenizer, text, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=5, max_length=128)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Train and Evaluate
train(model, train_dataloader, optimizer, device)
evaluate(model, val_dataloader, tokenizer, device)

# Example Translation
test_sentence = "እንዴት ነህ?"
print("Original:", test_sentence)
print("Translated:", translate(model, tokenizer, test_sentence, device))

# Save Model
model.save_pretrained("trained_amharic_english_model")
tokenizer.save_pretrained("trained_amharic_english_model")
