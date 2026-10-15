# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset processor for pre-tokenized datasets.

This processor passes through pre-tokenized data without re-tokenization,
preserving the exact input_ids from the dataset.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor


if TYPE_CHECKING:
    pass


logger = logging.get_logger(__name__)


@dataclass
class PreTokenizedDatasetProcessor(DatasetProcessor):
    """
    Dataset processor that passes through pre-tokenized data.

    Expects the dataset to already have:
        - input_ids: List of token IDs
        - attention_mask: List of attention mask values (optional, will be generated if missing)
        - labels: List of label IDs (optional, will copy input_ids if missing)
    """

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        Pass through pre-tokenized examples with minimal processing.

        Args:
            examples: Batched examples containing input_ids, and optionally
                     attention_mask and labels.

        Returns:
            Processed examples with input_ids, attention_mask, and labels.
        """
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        optional_passthrough_keys = [
            "position_ids",
            "segment_ids",
            "current_step_mask",
            "unresolved_mask",
            "step_map",
            "teacher_topk_ids",
            "teacher_topk_logprobs",
            "prefix_len",
            "block_len",
            "block_index",
            "step",
            "trajectory_prompt_len",
            "trajectory_response_len",
            "trajectory_step_map",
            "trajectory_block_size",
        ]
        for key in optional_passthrough_keys:
            if key in examples:
                model_inputs[key] = []

        for i in range(len(examples["input_ids"])):
            input_ids = examples["input_ids"][i]

            # Ensure input_ids is a list
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()

            # Get or generate attention_mask
            if "attention_mask" in examples and examples["attention_mask"][i] is not None:
                attention_mask = examples["attention_mask"][i]
                if hasattr(attention_mask, "tolist"):
                    attention_mask = attention_mask.tolist()
            else:
                # Generate attention mask (all 1s for non-padding tokens)
                attention_mask = [1] * len(input_ids)

            # Get or generate labels
            if "labels" in examples and examples["labels"][i] is not None:
                labels = examples["labels"][i]
                if hasattr(labels, "tolist"):
                    labels = labels.tolist()
            else:
                # Use input_ids as labels (standard causal LM training)
                labels = input_ids.copy()

            # Apply cutoff length if specified
            cutoff_len = self.data_args.cutoff_len
            if cutoff_len and len(input_ids) > cutoff_len:
                if self.data_args.truncate_mode == "drop":
                    # Skip this example if it's too long
                    continue
                else:
                    # Truncate to cutoff length
                    input_ids = input_ids[:cutoff_len]
                    attention_mask = attention_mask[:cutoff_len]
                    labels = labels[:cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
            for key in optional_passthrough_keys:
                if key not in examples:
                    continue
                value = examples[key][i]
                if hasattr(value, "tolist"):
                    value = value.tolist()
                model_inputs[key].append(value)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print a data example to stdout for debugging."""
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

        print("label_ids:\n{}".format(example["labels"]))
        valid_labels = [token_id if token_id != IGNORE_INDEX else self.tokenizer.pad_token_id for token_id in example["labels"]]
        print("labels:\n{}".format(self.tokenizer.decode(valid_labels, skip_special_tokens=False)))
