import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

CACHE_DIR = ".hf_cache_clean"


class BertWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR)
        config.torchscript = True
        if hasattr(config, "loss_type"):
            delattr(config, "loss_type")
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", config=config, cache_dir=CACHE_DIR
        )
        self.bert.eval()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )[0]


model = BertWrapper().eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR)
example = tokenizer("Hello world", return_tensors="pt")
input_ids = example["input_ids"]
attention_mask = example["attention_mask"]

with torch.no_grad():
    traced_model = torch.jit.trace(
        model, (input_ids, attention_mask), check_trace=False
    )
    traced_model.save("bert_libtorch.pt")
