import json
import yaml
import argparse
import pprint as pp


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dataset to LLM chat format.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--system-prompt', type=str, required=True, help='Path to the system prompt YAML file.')
    parser.add_argument('--prompt-version', type=str, required=True, help='Version of the prompt to use.')
    parser.add_argument('--errors-json', type=str, required=False, help='JSON file containing error type descriptions.')
    return parser.parse_args()

def convert_to_llm_chat(sample, system_prompt, prompt_version, error_types=None):
    prompt = system_prompt[prompt_version]['system']
    if "{gec_error_definitions}" in prompt:
        assert error_types is not None, "Error types must be provided if prompt contains error definitions."
        prompt = prompt.format(gec_error_definitions=pp.pformat(error_types))
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{sample['source']}"},
            {"role": "assistant", "content": f"{sample['correct']}"}
        ]
    }

if __name__ == "__main__":
    args = parse_args()
    with open(args.system_prompt, 'r') as f:
        system_prompt = yaml.safe_load(f)
    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f]
    if args.errors_json:
        with open(args.errors_json, 'r') as f:
            error_types = json.load(f)
    
    with open(args.output, 'w') as f:
        for item in data:
            f.write(json.dumps(convert_to_llm_chat(item,
                                                   system_prompt,
                                                   args.prompt_version,
                                                   error_types=None if not args.errors_json else error_types)) + '\n')