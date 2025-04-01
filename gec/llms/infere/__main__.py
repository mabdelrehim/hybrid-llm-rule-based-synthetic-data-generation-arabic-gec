import argparse
import torch
import yaml
import json
import random
import logging
import pprint as pp

random.seed(42)

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from peft import PeftModel
from camel_tools.tokenizers.word import simple_word_tokenize
from tqdm.auto import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--dataset', type=str, help='Dataset file in the format of a jsonl file each line containing 2 fields "source" and "correct"')
    parser.add_argument('--prompt', type=str, default=None, help='YAML file containing system prompt')
    parser.add_argument('--gec-error-definitions', type=str, default=None, help='Path to json containing error definitions for arabic gec')
    parser.add_argument('--prompt-version', type=str, default=None, help='prompt version')
    parser.add_argument('--output', type=str, help='Output file to write the predictions to')
    parser.add_argument('--model', type=str, help='Model name or path to model')
    parser.add_argument('--adapter', type=str, required=False, help='Peft Adapter')
    parser.add_argument('--n-shots', type=int, default=0, help='Number of shots to use for in-context learning')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size to use')
    parser.add_argument('--examples', type=str, default=None, help='Path to the examples file when n-shots is > 0')
    args = parser.parse_args()
    return args


def construct_prompt_messages(args):
    with open(args.prompt, "r") as f:
        prompt = yaml.safe_load(f)
        
    if args.gec_error_definitions:
        with open(args.gec_error_definitions, "r") as f:
            error_definitions = json.load(f)
        prompt_messages = [
            {"role": "system",
             "content": prompt[args.prompt_version]["system"].format(gec_error_definitions=error_definitions)}
        ]
    else:
        prompt_messages = [
            {"role": "system",
             "content": prompt[args.prompt_version]["system"]}
        ]
    
    if args.n_shots > 0:
        with open(args.examples, "r") as f:
            examples = [json.loads(line) for line in f]
        
        examples = examples[:args.n_shots]

        for example in examples:
            prompt_messages.append({"role": "user", "content": example["source"]})
            prompt_messages.append({"role": "assistant", "content": example["correct"]})
    
    return prompt_messages


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              padding_side='left',
                                              trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 device_map="cuda",
                                                 trust_remote_code=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model,
                                          args.adapter)
        model = model.merge_and_unload()
        model.eval()
        print(model)
    pipe = pipeline("text-generation",
                    model=args.model,
                    device_map="cuda",
                    tokenizer=tokenizer,
                    trust_remote_code=True)
    with open(args.dataset, "r") as f:
        dataset = [json.loads(line) for line in f]
    
    if args.prompt:
        prompt_messages = construct_prompt_messages(args)
    
    # output = []
    data = []
    for example in dataset:
        if args.prompt:
            messages = prompt_messages.copy()
            messages.append({"role": "user", "content": example["source"]})
        else:
            messages = [{"role": "user", "content": example["source"]}]
        data.append({"messages": messages})
        
    # with open(args.output, 'w') as fo:
    #     for out in tqdm(pipe(KeyDataset(data, "messages"), batch_size=args.batch_size, max_length=tokenizer.model_max_length)):
    #         result = out[0]['generated_text'][-1]["content"]
    #         result = " ".join(simple_word_tokenize(result.strip()))
    #         fo.write(result.strip() + '\n')
  
    with open(args.output, 'w') as fo:
        for messages in tqdm(data):
            prompt_output = pipe(messages["messages"], max_length=tokenizer.model_max_length)
            result = prompt_output[0]['generated_text'][-1]["content"]
            result = " ".join(simple_word_tokenize(result.strip()))
            fo.write(result.strip() + '\n')
        
    # with open(args.output, 'w') as fo:
    #     for l in output:
    #         l = " ".join(simple_word_tokenize(l.strip()))
    #         fo.write(l.strip() + '\n')
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
