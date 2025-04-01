import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a text files to jsonl')
    parser.add_argument('--input_source', type=str, help='Path to the input source file')
    parser.add_argument('--input_correct', type=str, help='Path to the input source file')
    parser.add_argument('--output', type=str, help='Path to the output file')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.input_source, 'r') as source_file, open(args.input_correct, 'r') as correct_file, open(args.output, 'w') as output_file:
        for source, correct in zip(source_file, correct_file):
            data = {
                'source': source.strip(),
                'correct': correct.strip()
            }
            output_file.write(json.dumps(data) + '\n')
            
if __name__ == '__main__':
    main()