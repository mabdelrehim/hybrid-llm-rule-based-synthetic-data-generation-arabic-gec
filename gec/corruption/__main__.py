import argparse
import random
import json
from .rule_based import RuleBasedCorruptor
from .hybrid_llm_rule_based import HybridLLMRuleBasedCorruptor
from .hybrid_gpt_rule_based import HybridGPTRuleBasedCorruptor
from tqdm.auto import tqdm

def write_buffer(path, current_buffer, last_processed_line):
    with open(path, 'a') as f:
        for line in current_buffer:
            f.write(json.dumps(line).strip() + '\n')
    with open(path.replace(".jsonl", ".checkpoint.json"), 'w') as f:
        f.write(json.dumps({'last_processed_line': last_processed_line}))
        
def chunks(L, n):
    """ Yield successive n-sized chunks from L.
    """
    for i in range(0, len(L), n):
        yield L[i:i+n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--errors-prior", type=str, default=None, required=False)
    parser.add_argument("--llm", type=str, default=None, required=False)
    parser.add_argument("--error-definitions", type=str, default=None, required=False)
    args = parser.parse_args()
    
    if args.errors_prior:
        with open(args.errors_prior) as f:
            errors_prior = json.load(f)
    else:
        errors_prior = None

    if args.llm is not None:
        if args.llm == "gpt-4o" or  args.llm == "gpt-4o-mini":
            corrupt = HybridGPTRuleBasedCorruptor(tag_distribution=errors_prior,
                                                  model_name=args.llm,
                                                  error_description=args.error_definitions)
        else:
            corrupt = HybridLLMRuleBasedCorruptor(tag_distribution=errors_prior,
                                                  model_name=args.llm,
                                                  error_description=args.error_definitions)
    else:
        corrupt = RuleBasedCorruptor(tag_distribution=errors_prior)
    
    with open(args.input) as f:
        lines = f.readlines()

    if args.clear:
        assert not args.checkpoint, "invalid arguments"
        # clear the data in the info file
        with open(args.output,'w') as file:
            pass
    if args.checkpoint:
        with open(args.checkpoint) as f:
            checkpoint = json.load(f)['last_processed_line']
    else:
        checkpoint = -1
    
    current_buffer = []
    last_processed_line = 0
    random.shuffle(lines)
    i = 0
    with tqdm(total=len(lines)) as pbar:
        for batch in chunks(lines, args.batch_size):
            if i > checkpoint:
                out_batch = corrupt([l.strip() for l in batch])
                current_buffer.extend(out_batch)
                if len(current_buffer) % (args.batch_size * 1) == 0:
                    write_buffer(args.output, current_buffer, i)
                    current_buffer = []
            i += len(batch) - 1 # zero indexed
            last_processed_line = i
            pbar.update(len(batch))
    if len(current_buffer) > 0:
        write_buffer(args.output, current_buffer, last_processed_line)
        current_buffer = []

        
    
if __name__ == "__main__":
    main()
