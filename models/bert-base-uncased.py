import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

class BertWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        if hasattr(config, "loss_type"):
            delattr(config, "loss_type")
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

model = BertWrapper()
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
example = tokenizer("Hello world", return_tensors="pt")
input_ids = example["input_ids"]
attention_mask = example["attention_mask"]

traced_model = torch.jit.trace(model, (input_ids, attention_mask))
traced_model.save("bert_libtorch.pt")