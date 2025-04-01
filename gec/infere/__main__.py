import argparse
import torch
import logging
import string
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from tqdm.auto import tqdm
from  camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--no-punct', action="store_true")
    parser.add_argument('--decoding', type=str, required=False, choices=["beam", "top_p"], default="beam")
    return parser.parse_args()

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET)) + '&amp;'
pnx_re = re.compile(r'([' + re.escape(puncs) + '])')
space_re = re.compile(' +')

def remove_punct(line):
    line = line.strip()
    line = pnx_re.sub(r'', line)
    line = space_re.sub(' ', line)
    line = line.strip()
    return line


def main():
    args = parse_args()
    logging.info(f"TEST DATA: {args.test_data}")
    logging.info(f"OUTPUT FILE: {args.output_file}")
    logging.info(f"MODEL: {args.model}")
    logging.info(f"TOKENIZER: {args.tokenizer}")
    logging.info(f"BATCH SIZE: {args.batch_size}")
    logging.info(f"DEVICE: {args.device}")
    logging.info(f"NO PUNCT: {args.no_punct}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    test_data = load_dataset("json", data_files={"test": args.test_data})
    if args.decoding == "top_p":
        gen_kwargs = {"max_length": 1024,
                      "num_beams": 5,
                      "decoder_start_token_id": 0,
                      "top_p": 0.8,
                      "top_k": 75,
                      "temperature": 0.8,
                      "do_sample": True}
    elif args.decoding == "beam":
        gen_kwargs = {"max_length": 1024,
                      "num_beams": 5,
                      "decoder_start_token_id": 0}
    else:
        raise ValueError(f"Invalid decoding strategy {args.decoding}")

    def preprocess_function(examples):
        if args.no_punct:
            inputs = [remove_punct(ex) for ex in examples["source"]]
            targets = [remove_punct(ex) for ex in examples["correct"]]
        else:
            inputs = [ex for ex in examples["source"]]
            targets = [ex for ex in examples["correct"]]

        model_inputs = tokenizer(text=inputs,
                                 max_length=1024,
                                 truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets,
                           max_length=1024,
                           truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def postprocess_text(preds):
        if args.no_punct:
            preds = [remove_punct(pred).strip() for pred in preds]
        else:
            preds = [pred.strip() for pred in preds]
        return preds
    
    column_names = test_data["test"].column_names
    processed_datasets = test_data.map(preprocess_function,
                                       batched=True,
                                       num_proc=2,
                                       remove_columns=column_names,
                                       desc="Running tokenizer on dataset")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model,
                                           label_pad_token_id=-100)
    test_data = processed_datasets["test"]
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  shuffle=False,
                                                  collate_fn=data_collator,
                                                  batch_size=args.batch_size,
                                                  num_workers=6)
    
    model_outs = []
    for step, batch in tqdm(enumerate(test_dataloader)):
        batch.to(args.device)
        with torch.no_grad():
          generated_tokens = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
          )
          decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy(), skip_special_tokens=True)
          decoded_preds = postprocess_text(decoded_preds)
          model_outs.extend(list(decoded_preds))

    with open(args.output_file, 'w') as fo:
      for l in model_outs:
        l = " ".join(simple_word_tokenize(l.strip()))
        fo.write(l.strip() + '\n')
        
if __name__ == "__main__":
    main()