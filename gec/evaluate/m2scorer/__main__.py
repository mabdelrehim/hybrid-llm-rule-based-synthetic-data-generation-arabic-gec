from gec.evaluate.m2scorer import evaluate, evaluate_single_sentences
import argparse


def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--system_output')
    parser.add_argument('--m2_file')
    parser.add_argument('--mode', default='all')

    args = parser.parse_args()

    if args.mode == 'all':
        evaluate(args.system_output, args.m2_file, timeout=30)
    else:
        evaluate_single_sentences(args.system_output, args.m2_file, timeout=120)
        
if __name__ == "__main__":
    main()