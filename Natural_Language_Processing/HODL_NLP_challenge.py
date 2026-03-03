from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset

# Do not change function signature
def preprocess_function(examples: dict) -> dict:
    # Input:  {"text": List[str], "label": List[int]} 
    # Output: {"input_ids": [B, L], "attention_mask": [B, L]} 
    
    # Initialize tokenizer (using a pretrained BERT tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # tokenize and truncate + pad to max_length=256
    tokenized_sample = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length', 
        max_length=512,
        return_tensors=None
    )
    return tokenized_sample


# Do not change function signature
def init_model() -> nn.Module:
    # Input dimension:  [B, L]   (B = batch size, L = sequence length of tokenized review)
    # Output dimension: [B, 2]   (logits for sentiment classes: negative / positive)
    
    ''' 0.91
    model = AutoModelForSequenceClassification.from_pretrained(
      'nlptown/bert-base-multilingual-uncased-sentiment',  # Model trained on reviews
      num_labels=2
    )
    '''
    
    model = AutoModelForSequenceClassification.from_pretrained(
      'roberta-base',
      num_labels=2
    )
    return model


# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
    # dev_dataset: tokenized IMDB dataset
    # Each sample: 
    #   input_ids: [L]        (token IDs for a review)
    #   attention_mask: [L]   (mask for padding)
    #   labels: int (0 = negative, 1 = positive)
    
    # training args
    training_args = TrainingArguments(
      output_dir='./results',
      num_train_epochs=4,  # Increased epochs
      per_device_train_batch_size=8,  # Smaller batch size for better generalization
      learning_rate=1e-5,  # Lower learning rate for fine-tuning
      warmup_ratio=0.1,  # Use ratio instead of fixed steps
      weight_decay=0.01,
      logging_dir='./logs',
      logging_steps=50,
      save_strategy='no',
      evaluation_strategy='no',
      gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
      fp16=True,  # Mixed precision for faster training
    )
    
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=dev_dataset,
    )
    
    trainer.train()
    
    return model

