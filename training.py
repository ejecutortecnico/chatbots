import torch
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset

class CustomTrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        item = df.iloc[idx]
        text = item['text']
        label = item['label']
        # encode text
        encoding = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        # remove batch dimension which the tokenizer automatically adds
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # add label
        encoding["label"] = torch.tensor(label)
        
        return encoding

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
df = pd.read_csv("path_to_your_csv")

train_dataset = CustomTrainDataset(df=df, tokenizer=tokenizer)


train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
batch = next(iter(train_dataloader))
for k,v in batch.items():
    print(k, v.shape)
# decode the input_ids of the first example of the batch
print(tokenizer.decode(batch['input_ids'][0].tolist()))
# Instantiate pre-trained BERT model with randomly initialized classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# I almost always use a learning rate of 5e-5 when fine-tuning Transformer based models
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# put model on GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        # put batch on device
        batch = {k:v.to(device) for k,v in batch.items()}
        
        # forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            # put batch on device
            batch = {k:v.to(device) for k,v in batch.items()}
            
            # forward pass
            outputs = model(**batch)
            loss = outputs.logits
            
            val_loss += loss.item()
                  
    print("Validation loss after epoch {epoch}:", val_loss/len(eval_dataloader))