import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, BertPreTrainedModel, BertModel, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset
from TorchCRF import CRF
# Set environment variable to handle tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BertForTokenClassificationCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.num_tags = config.num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        if labels is not None:
            labels = labels.to(device=self.crf.start_transitions.device, dtype=torch.long)
            active_labels = torch.where(labels == -100, torch.full_like(labels, 0), labels)
            mask = attention_mask.byte() if attention_mask is not None else None
            loss = -self.crf(logits, active_labels, mask=mask, reduction='mean')
            return loss, logits
        else:
        # This is crucial for ensuring correct decoding during evaluation
            return logits, self.crf.decode(logits, mask=attention_mask.byte())




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
            else:  # empty line indicates the end of a sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)
    return sentences, tags

def tokenize_and_align_labels(sentences, tags, tokenizer, tag2id):
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=True, return_tensors="pt")
    labels = []
    for i, label_list in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else tag2id.get(label_list[word_id], -100) for word_id in word_ids]
        labels.append(label_ids)
    tokenized_inputs["labels"] = torch.tensor(labels)
    return tokenized_inputs

class POSDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(pred):
    labels = pred.label_ids  # True labels (list of lists of ints)
    preds = pred.predictions  # Predicted labels from CRF decode (list of lists of ints)

    # Flatten the list of lists for both labels and predictions
    true_entities = [label for sentence in labels for label in sentence if label != -100]
    pred_entities = [pred for sentence, label in zip(preds, labels) for pred, lab in zip(sentence, label) if lab != -100]

    # Ensure there are no unexpected float types
    true_entities = [int(x) for x in true_entities]
    pred_entities = [int(x) for x in pred_entities]

    precision, recall, f1, _ = precision_recall_fscore_support(true_entities, pred_entities, average='weighted', zero_division=0)
    acc = accuracy_score(true_entities, pred_entities)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def main():
    train_tags = collect_unique_tags('train.txt')
    validation_tags = collect_unique_tags('validation.txt') if os.path.exists('validation.txt') else set()

    all_tags = train_tags.union(validation_tags)
    print("All unique tags:", all_tags)

    tag_set = sorted(list(all_tags))
    tag2id = {tag: idx for idx, tag in enumerate(tag_set)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    model = BertForTokenClassificationCRF.from_pretrained('bert-base-multilingual-cased', num_labels=len(tag_set))
    model.config.id2label = id2tag
    model.config.label2id = tag2id
    model.config.num_labels = len(tag_set)

    train_sentences, train_tags = read_train_data('train.txt')
    validation_sentences, validation_tags = read_train_data('validation.txt')

    train_encodings = tokenize_and_align_labels(train_sentences, train_tags, tokenizer, tag2id)
    validation_encodings = tokenize_and_align_labels(validation_sentences, validation_tags, tokenizer, tag2id)

    train_dataset = POSDataset(train_encodings)
    validation_dataset = POSDataset(validation_encodings)

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
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == '__main__':
    main()
