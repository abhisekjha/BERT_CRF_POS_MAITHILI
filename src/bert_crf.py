import os
import torch
from torch import nn
from TorchCRF import CRF
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Set environment variable to handle tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collect_unique_tags(filename):
    unique_tags = set()
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    tag = parts[1].strip()
                    unique_tags.add(tag)
    return unique_tags

# Collect tags
train_tags = collect_unique_tags('datasets/train.txt')
validation_tags = collect_unique_tags('datasets/validation.txt') if os.path.exists('datasets/validation.txt') else set()
all_tags = train_tags.union(validation_tags)
tag_set = sorted(list(all_tags))
tag2id = {tag: idx for idx, tag in enumerate(tag_set)}
id2tag = {idx: tag for tag, idx in tag2id.items()}

class BertCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertCRF, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.crf = CRF(num_labels)  # Ensure this is the correct CRF import

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            # Make sure loss is a scalar
            loss = -self.crf(logits, labels, mask=attention_mask.byte()) if attention_mask is not None else -self.crf(logits, labels)
            loss = loss.mean()  # Ensure loss is scalar if necessary
            return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            return self.crf.decode(logits, mask=attention_mask.byte()) if attention_mask is not None else self.crf.decode(logits)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
model = BertCRF('bert-base-multilingual-cased', num_labels=len(tag_set))
model.to(device)

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

train_sentences, train_tags = read_train_data('datasets/train.txt')
validation_sentences, validation_tags = read_train_data('datasets/validation.txt')

def tokenize_and_align_labels(sentences, tags):
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=True, return_tensors="pt")
    labels = []
    for i, label_list in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else tag2id.get(label_list[word_id], -100) for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = torch.tensor(labels).to(device)
    return tokenized_inputs

train_encodings = tokenize_and_align_labels(train_sentences, train_tags)
validation_encodings = tokenize_and_align_labels(validation_sentences, validation_tags)

class POSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = POSDataset(train_encodings)
validation_dataset = POSDataset(validation_encodings)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    true_entities = [label for sentence in labels for label in sentence if label != -100]
    pred_entities = [pred for sentence, label in zip(preds, labels) for pred, lab in zip(sentence, label) if lab != -100]
    precision, recall, f1, _ = precision_recall_fscore_support(true_entities, pred_entities, average='weighted', zero_division=0)
    acc = accuracy_score(true_entities, pred_entities)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
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
