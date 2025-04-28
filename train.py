from data import train_dataset, test_dataset
from model import czytacz, processor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import evaluate
import numpy as np

wer = evaluate.load('wer')
cer = evaluate.load('cer')
exact_match = evaluate.load("exact_match")


def compute_metrics(pred):
    preds = pred.predictions
    labels = pred.label_ids
    preds = np.argmax(preds[0], axis=-1)

    labels[labels == -100] = processor.tokenizer.pad_token_id

    # print(preds)
    # print(labels)

    pred_texts = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(pred_texts)
    # print(label_texts)

    wer_score = wer.compute(predictions=pred_texts, references=label_texts)
    cer_score = cer.compute(predictions=pred_texts, references=label_texts)
    exact_match_score = exact_match.compute(predictions=pred_texts, references=label_texts)['exact_match']

    return {"wer": wer_score, "cer": cer_score, 'exact_match': exact_match_score}


training_args = Seq2SeqTrainingArguments(
    output_dir="czytacz",

    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,
    weight_decay=0.01,
    bf16=True,

    lr_scheduler_type="cosine",
    warmup_steps=100,

    logging_strategy='steps',
    logging_steps=50,

    label_names=["labels"],

    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    push_to_hub=True,
)


early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.005,
)


trainer = Seq2SeqTrainer(
    model=czytacz,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

trainer.train()

czytacz.save_pretrained("czytacz")
processor.save_pretrained("czytacz")
