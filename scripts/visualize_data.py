import json
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=10, required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    data = []
    with open(args.input_file, 'r') as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    for i in range(args.num_samples):
        sample = random.choice(data)
        print("source: ", sample['source'])
        print("correct: ", sample['correct'])
        print()
        
if __name__ == '__main__':
    main()
