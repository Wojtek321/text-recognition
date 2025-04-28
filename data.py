from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BlipProcessor


class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        text = item['text']

        inputs = self.processor(image, padding="max_length", return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)

        encoding = self.processor.tokenizer(text, truncation=True, padding='max_length', max_length=128,
                                            return_tensors='pt')
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs['input_ids'] = input_ids
        inputs['attention_mask'] = attention_mask
        inputs['labels'] = labels
        # print(inputs)

        return inputs


def load_iam_dataset():
    train_dataset = load_dataset("Teklia/IAM-line", split="train")

    train_dataset = IAMLineDataset(
        train_dataset,
        processor
    )

    test_dataset= load_dataset("Teklia/IAM-line", split="test[:25%]")

    test_dataset = IAMLineDataset(
        test_dataset,
        processor
    )

    print(f"Dataset loaded with {len(train_dataset)} training samples and {len(test_dataset)} test samples")

    return train_dataset, test_dataset


def get_data_loaders(train_dataset=None, test_dataset=None, batch_size=16):
    train_dataloader = None
    test_dataloader = None

    if train_dataset:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False
        )

    if test_dataset:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False
        )

    return train_dataloader, test_dataloader


model_name = 'Salesforce/blip-image-captioning-base'
processor = BlipProcessor.from_pretrained(model_name, use_fast=True)


if __name__ == '__main__':
    train_dataset, test_dataset = load_iam_dataset()
    a = train_dataset[0]
    print(type(a))
    print({k: v.shape if hasattr(v, 'shape') else v for k, v in a.items()})
