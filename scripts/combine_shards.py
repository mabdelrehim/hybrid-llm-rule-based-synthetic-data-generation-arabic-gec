import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_files", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.output_file, "w") as out_file:
        for shard_file in args.shard_files:
            print(f"adding {shard_file}")
            with open(shard_file) as in_file:
                for line in in_file:
                    line = json.loads(line.strip())
                    out_file.write(json.dumps({'source': line['incorrect'],
                                               'correct': line['correct']}) + "\n")

if __name__ == "__main__":
    main()