from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None,
                                       metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    use_fast_tokenizer: bool = field(default=True,
                                     metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},)


@dataclass
class GECTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_key: str = field(default="source",
                            metadata={"help": "Source language id for translation."})
    target_key: str = field(default="correct",
                            metadata={"help": "Target language id for translation."})
    train_file: Optional[str] = field(default=None,
                                      metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(default=None,
                                           metadata={"help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."},)
    preprocessing_num_workers: Optional[int] = field(default=None,
                                                     metadata={"help": "The number of processes to use for the preprocessing."},)
    max_source_length: Optional[int] = field(default=512,
                                             metadata={"help": ("The maximum total input sequence length after tokenization. Sequences longer "
                                                                "than this will be truncated, sequences shorter will be padded.")},)
    max_target_length: Optional[int] = field(default=512,
                                             metadata={"help": ("The maximum total sequence length for target text after tokenization. Sequences longer "
                                                                "than this will be truncated, sequences shorter will be padded.")},)
    num_beams: Optional[int] = field(default=2,
                                     metadata={"help": ("Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                                                        "which is used during ``evaluate`` and ``predict``.")},)
    num_return_sequences: Optional[int] = field(default=1,
                                                metadata={"help": ("Number of generated sequences. This argument will be passed to ``model.generate``, "
                                                                   "which is used during ``evaluate`` and ``predict``.")},)
    ignore_pad_token_for_loss: bool = field(default=True,
                                            metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},)
    
    