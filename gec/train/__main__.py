import logging
import numpy as np

from transformers import (
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments,
  GenerationConfig,
  DataCollatorForSeq2Seq,
  set_seed,
  HfArgumentParser,
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  EvalPrediction,
  T5ForConditionalGeneration
)

from datasets import load_dataset
from tqdm.auto import tqdm
from sacrebleu import corpus_bleu, corpus_chrf

from gec.train.args import ModelArguments, GECTrainingArguments


logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, GECTrainingArguments, Seq2SeqTrainingArguments))
    model_args, gec_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                              use_fast=model_args.use_fast_tokenizer)
    
    def preprocess_function(examples):
        inputs = [ex for ex in examples[gec_args.source_key]]
        targets = [ex for ex in examples[gec_args.target_key]]

        model_inputs = tokenizer(text=inputs,
                                 max_length=gec_args.max_source_length,
                                 truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets,
                           max_length=gec_args.max_target_length,
                           truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def postprocess_text(preds, labels=None):
      preds = [pred.strip() for pred in preds]
      labels = [[label.strip()] for label in labels]
      return preds, labels
    
    def compute_metrics(p: EvalPrediction):
        if gec_args.ignore_pad_token_for_loss:
          # Replace -100 in the labels as we can't decode them.
          p.label_ids = np.where(p.label_ids != -100, p.label_ids, tokenizer.pad_token_id)
        preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in p.predictions]
        labels = [tokenizer.decode(label, skip_special_tokens=True) for label in p.label_ids]
        preds, labels = postprocess_text(preds, labels)
        bleu_score = corpus_bleu(preds, labels).score
        chrf_score = corpus_chrf(preds, labels).score
        return {"bleu": bleu_score, "chrf": chrf_score}
    
    if training_args.do_train:
        raw_data = load_dataset("json",
                              data_files={
                                "train": gec_args.train_file,
                                "dev": gec_args.validation_file
                              })
        # Preprocessing.
        column_names = raw_data["train"].column_names
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            preprocessed_datasets = raw_data.map(preprocess_function,
                                                batched=True,
                                                num_proc=gec_args.preprocessing_num_workers,
                                                remove_columns=column_names,
                                                load_from_cache_file=False,
                                                desc="Running tokenizer on data",)
        train_dataset = preprocessed_datasets["train"]
        eval_dataset = preprocessed_datasets["dev"]
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                               model=model,
                                               label_pad_token_id=-100, # ignore padding in loss calculation
                                               pad_to_multiple_of=8 if training_args.fp16 else None)
        training_args.generation_config = GenerationConfig(max_length=gec_args.max_target_length,
                                                           num_beams=gec_args.num_beams,
                                                           decoder_start_token_id=model.generation_config.decoder_start_token_id,
                                                           bos_token_id=model.generation_config.bos_token_id,
                                                           eos_token_id=tokenizer.eos_token_id,
                                                           pad_token_id=tokenizer.pad_token_id)
        trainer = Seq2SeqTrainer(model=model,
                                 args=training_args,
                                 data_collator=data_collator,
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 tokenizer=tokenizer,
                                 compute_metrics=compute_metrics)

        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
if __name__ == "__main__":
    main()
