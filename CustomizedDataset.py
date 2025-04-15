from torch.utils.data import Dataset

class myDataset(Dataset):
  def __init__(self, file_path, tokenizer, max_length=128):
    self.tokenizer=tokenizer
    self.max_length=max_length

    with open(file_path, "r", encoding="utf-8") as file:
      self.texts = [line.strip() for line in file.readlines()]

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = self.texts[item]

    encoding = self.tokenizer(
      text,
      add_special_tokens=True,
      truncation=True,
      padding="max_length",
      max_length=self.max_length,
      return_tensors="pt"
    )

    return {
      "input_ids": encoding["input_ids"].squeeze(0),
      "attention_mask": encoding["attention_mask"].squeeze(0),
      "labels": encoding["input_ids"].squeeze(0)
    }
