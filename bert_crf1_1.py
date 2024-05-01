import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, BertPreTrainedModel, BertModel, Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from TorchCRF import CRF

# Configuration
class Config:
    train_file = 'datasets/train1.txt'
    validation_file = 'datasets/validation1.txt'
    model_name = 'bert-base-multilingual-cased'
    output_dir = './results'
    max_length = 128  # Maximum input length
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-5
    weight_decay = 0.01

# Utils
def collect_unique_tags(filename):
    unique_tags = set()
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        tag = parts[1].strip()
                        unique_tags.add(tag)
    except FileNotFoundError:
        print(f"File not found: {filename}")
    return unique_tags

def read_train_data(filename):
    sentences, tags = [], []
    with open(filename, 'r', encoding='utf-8') as file:
        current_sentence = []
        current_tags = []
        for line in file:
            line = line.strip()
            if line:
                word, tag = line.split('\t')
                current_sentence.append(word)
                current_tags.append(tag)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)
    return sentences, tags

# Custom Dataset
class POSDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# Model
class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            return logits, self.crf.decode(logits, mask=attention_mask.byte())

# Setup
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
model = BertCRF.from_pretrained(config.model_name, num_labels=len(tag_set))
model.to(device)

# Data preparation
train_sentences, train_tags = read_train_data(config.train_file)
validation_sentences, validation_tags = read_train_data(config.validation_file)

train_encodings = tokenize_and_align_labels(train_sentences, train_tags, tokenizer, tag2id)
validation_encodings = tokenize_and_align_labels(validation_sentences, validation_tags, tokenizer, tag2id)

train_dataset = POSDataset(train_encodings)
validation_dataset = POSDataset(validation_encodings)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

# Training
training_args = TrainingArguments(
    output_dir=config.output_dir,
    evaluation_strategy="epoch",
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=config.num_epochs,
    weight_decay=config.weight_decay,
    logging_dir='./logs',
    logging_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
