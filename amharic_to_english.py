# import torch
# import requests
# import os
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from totils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from tqdm import tqdm

# # 1. Download Files
# file_urls = {
#     "amharic_bible.txt": "https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_E-Bible/amharic.txt",
#     "english_bible.txt": "https://raw.githubusercontent.com/MarsPanther/Amharic-English-Machine-Translation-Corpus/master/Amharic_English_E-Bible/english.txt"
# }

# def download_files(urls, directory="./data"):
#     os.makedirs(directory, exist_ok=True)
#     for filename, url in urls.items():
#         filepath = os.path.join(directory, filename)
#         print(f"Downloading {url} to {filepath}")
#         try:
#             response = requests.get(url, stream=True)
#             response.raise_for_status()
#             with open(filepath, 'wb') as file:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     file.write(chunk)
#         except requests.exceptions.RequestException as e:
#             print(f"Error downloading {url}: {e}")
#             exit()
#     print("Downloaded all files")

# download_files(file_urls)

# # Check GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 2. Load and Combine Datasets
# def load_texts(am_file, en_file):
#     with open(am_file, "r", encoding="utf-8") as am_f, open(en_file, "r", encoding="utf-8") as en_f:
#         amharic_data = am_f.readlines()
#         english_data = en_f.readlines()
#     return amharic_data, english_data

# # Load dataset
# amharic_sentences, english_sentences = load_texts("./data/amharic_bible.txt", "./data/english_bible.txt")

# # Ensure dataset sizes match
# assert len(amharic_sentences) == len(english_sentences), "Mismatch in dataset sizes!"

# # 3. Tokenization
# tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

# def tokenize_data(amharic, english):
#     return tokenizer(amharic, padding=True, truncation=True, return_tensors="pt"), tokenizer(english, padding=True, truncation=True, return_tensors="pt")

# amharic_encodings, english_encodings = tokenize_data(amharic_sentences, english_sentences)

# # 4. Load Model in PyTorch Format (No safetensors)
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M", from_tf=False, use_safetensors=False)
# model.to(device)

# # Define Dataset
# class TranslationDataset(Dataset):
#     def __init__(self, amharic_file, english_file, tokenizer, max_length=128):
#         self.amharic_lines = open(amharic_file, 'r', encoding='utf-8').readlines()
#         self.english_lines = open(english_file, 'r', encoding='utf-8').readlines()
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.amharic_lines)

#     def __getitem__(self, idx):
#         am_text = self.amharic_lines[idx].strip()
#         en_text = self.english_lines[idx].strip()

#         am_tokens = self.tokenizer(am_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
#         en_tokens = self.tokenizer(en_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

#         return {
#             'input_ids': am_tokens['input_ids'].squeeze(),
#             'attention_mask': am_tokens['attention_mask'].squeeze(),
#             'labels': en_tokens['input_ids'].squeeze()
#         }

# # Load Dataset
# dataset = TranslationDataset("./data/amharic_bible.txt", "./data/english_bible.txt", tokenizer)

# # Create DataLoader
# train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # 5. Training Setup
# optimizer = AdamW(model.parameters(), lr=5e-5)

# def train(model, dataloader, optimizer, device, epochs=3):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
#         for batch in progress_bar:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()

#             loss.backward()
#             optimizer.step()

#             progress_bar.set_postfix({"loss": loss.item()})

#         print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

# # Train
# train(model, train_dataloader, optimizer, device)

# # 6. Save PyTorch Model
# os.makedirs("trained_model", exist_ok=True)
# model.save_pretrained("trained_model")
# tokenizer.save_pretrained("trained_model")

# # 7. Evaluation Function
# def evaluate(model, dataloader, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()

#     print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")

# evaluate(model, train_dataloader, device)

# # 8. Translation Function
# def translate(model, tokenizer, text, device):
#     model.eval()
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)

#     with torch.no_grad():
#         generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=128)

#     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# # Example translation
# test_sentences = ["እንዴት ነህ?", "በህግ የተፈቀደ", "ዳኝነት", "ደንብ፣ ህግ፣ ሕግ፣ ሥርዓት"]
# for sentence in test_sentences:
#     translated = translate(model, tokenizer, sentence, device)
#     print(f"Original: {sentence}")
#     print(f"Translated: {translated}\n")
