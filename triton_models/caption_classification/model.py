import dataclasses
import json
import multiprocessing as mp
import pathlib
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import triton_python_backend_utils as pb_utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

TOKENIZER_NAME = "microsoft/deberta-large"


def _parse_param(params: Mapping, key: str):
    """Parse the model parameters from `model_config["parameters"]`.

    model_config["parameters"] is a dictionary containing the model parameters in this format:
    ```json
    {
        "threshold": {"string_value": "0.5"},
        "min_words": {"string_value": "5"}
    }
    ```
    """
    return params[key]["string_value"]


@dataclasses.dataclass
class ModelParams:
    """Triton model parameters.

    Attributes:
        threshold (float): The threshold value used for classification.
        min_words (int): The minimum number of words required for a caption to be considered valid.
    """

    threshold: float
    min_words: int

    @classmethod
    def from_dict(cls, d: Mapping):
        threshold = float(_parse_param(params=d, key="threshold"))
        min_words = int(_parse_param(params=d, key="min_words"))
        return cls(threshold=threshold, min_words=min_words)


@dataclasses.dataclass
class DataParams:
    """Triton model data processing parameters.

    Attributes:
        num_workers (int): The number of workers for data loading.
        batch_size (int): The batch size for data loading.
        max_len (int, optional): The maximum length of the input data. Defaults to 512.
    """

    num_workers: int
    batch_size: int
    max_len: int = 512


class CaptionCandidateDataset(Dataset):
    """
    Dataset class for caption candidate data.

    Args:
        items (np.ndarray): Array of items.
        tokenizer: Tokenizer object.
        max_len (int): Maximum length of the input sequence.
    """

    @dataclasses.dataclass
    class Item:
        input_ids: torch.Tensor
        attention_mask: torch.Tensor
        token_type_ids: torch.Tensor

    def __init__(
        self,
        items: np.ndarray,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        max_len: int,
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text = str(self.items[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        item = self.Item(
            input_ids=torch.tensor(inputs["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(inputs["attention_mask"], dtype=torch.long),
            token_type_ids=torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        )
        return dataclasses.asdict(item)


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

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.deberta_model(ids, attention_mask=mask, token_type_ids=token_type_ids)["last_hidden_state"][:, 0, :]
        output = self.ln(emb)
        output = self.out(output)
        return output


class TritonPythonModel:
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_config = json.loads(args["model_config"])
        self.model_params = ModelParams.from_dict(self.model_config["parameters"])
        self.data_params = DataParams(
            batch_size=self.model_config["max_batch_size"],
            num_workers=min(mp.cpu_count(), 8),
        )

        self._model_dir = pathlib.Path(args["model_repository"]) / args["model_version"]
        self.models = self._load_models()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def _get_output_dtype(self, output_name):
        """A helper function to get the output data type from the model configuration."""
        return pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, output_name)["data_type"]
        )

    def _load_models(self):
        weights_dir = self._model_dir / "weights"
        checkpoint_paths = sorted(weights_dir.rglob("*.pt"))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"No checkpoint files found in {weights_dir}")

        models = []
        for ckpt_path in checkpoint_paths:
            model = DebertaLargeModel()
            model = model.to(self.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            if self.device == "cuda":
                model = torch.nn.DataParallel(model)
            model.eval()
            models.append(model)

        return models

    def _make_data_loader(self, text_ls: np.ndarray) -> DataLoader:
        return DataLoader(
            dataset=CaptionCandidateDataset(
                items=text_ls,
                tokenizer=self.tokenizer,
                max_len=self.data_params.max_len,
            ),
            batch_size=self.data_params.batch_size,
            shuffle=False,
            num_workers=self.data_params.num_workers,
        )

    def _predict(self, candidates_stack: np.ndarray) -> np.ndarray:
        """For each group of caption candidates, predict the most likely candidate.

        `candidates_stack` in the form of
        ```
        [
            [caption1, caption2, ...],
            [caption3, caption4, ...],
            ...
        ]
        ```
        will first be exploded into a dataframe before being tokenized and fed into the model.
        ```
        image_id, text
        0, caption1
        0, caption2
        1, caption3
        1, caption4
        ```
        """
        df = pd.DataFrame(data=enumerate(candidates_stack), columns=["image_id", "text"])
        df = df.explode("text")

        # Preprocessing. Reduce the number of candidates:
        #   * Remove candidates with less than `min_words` words
        word_counts = df["text"].apply(lambda x: len(x.split()))
        df = df[(word_counts >= self.model_params.min_words)]

        # Prediction. Tokenize the text and feed it into the model to get predictions
        if len(df) > 0:
            data_loader = self._make_data_loader(df["text"].values)
            preds_all = []
            with torch.no_grad():
                for _item in data_loader:
                    item = CaptionCandidateDataset.Item(**_item)
                    input_ids = item.input_ids.to(self.device)
                    mask = item.attention_mask.to(self.device)
                    token_type_ids = item.token_type_ids.to(self.device)
                    preds = torch.stack([model(input_ids, mask, token_type_ids) for model in self.models])
                    preds_all.append(preds)
            df["pred"] = torch.stack(preds_all).squeeze().mean(0).detach().cpu().numpy()

            # Postprocessing.
            #   * Remove predictions below the threshold
            #   * Group by image_id and select the candidate with the highest prediction
            #       probability
            df = df[df["pred"] >= self.model_params.threshold]
            df = df.groupby("image_id").apply(lambda gr: gr.nlargest(1, "pred")).reset_index(drop=True)

        # Return the predictions in the original order. If there are no predictions,
        # return an empty string.
        caption_lookup = dict(zip(df["image_id"], df["text"]))
        return np.array([caption_lookup.get(image_id, "") for image_id in range(len(candidates_stack))])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        response = []
        for req in requests:
            candidates_stack = pb_utils.get_input_tensor_by_name(req, "candidates").as_numpy().astype(np.str_)
            predictions = self._predict(candidates_stack)
            caption_dtype = self._get_output_dtype("caption")
            caption_tensor = pb_utils.Tensor("caption", predictions.astype(caption_dtype))
            response.append(pb_utils.InferenceResponse(output_tensors=[caption_tensor]))
        return response
