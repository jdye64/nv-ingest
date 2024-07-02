import torch.nn as nn
import transformers

TOKENIZER_NAME = "microsoft/deberta-large"


class DebertaLargeModel(nn.Module):
    """The main model class."""

    MODEL_NAME = "microsoft/deberta-large"

    def __init__(self):
        super().__init__()
        self.deberta_model = transformers.DebertaModel.from_pretrained(
            self.MODEL_NAME,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            hidden_act="gelu_new",
        )
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, input_ids, input_mask, input_token_type_ids):
        # pooler
        emb = self.deberta_model(input_ids, attention_mask=input_mask, token_type_ids=input_token_type_ids)[
            "last_hidden_state"
        ][:, 0, :]
        output = self.ln(emb)
        output = self.out(output)
        return output
