# hybrid-llm-rule-based-synthetic-data-generation-arabic-gec
This repository contains the code and data for our paper Hybrid LLM and Rule-Based Synthetic Data Generation for Arabic Grammatical Error Correction accepted at ICMISI 2025.

## Set Up Environment (Ubuntu)

### Pre-installations
```bash
$ sudo apt-get install cmake libboost-all-dev
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Requirements
```bash
$ conda create -n arabic-gec python=3.10
$ conda activate arabic-gec
$ pip install -r requirements.txt
```

Note: If you are facing problems with camel-tools you might need to install it from source and comment out these lines from `setup.py` since we do not need this component in our work

```python
INSTALL_REQUIRES_NOT_WINDOWS = [
    'camel-kenlm >= 2023.3.17.2 ; platform_system!="Windows"'
]

if sys.platform != 'win32':
    INSTALL_REQUIRES.extend(INSTALL_REQUIRES_NOT_WINDOWS)
```

### CAMEL Models
```bash
$ camel_data -i all
```

### Areta

Please install using a separate environment

### Requirements
```bash
$ conda create -n arabic-gec-areta python=3.7
$ conda activate arabic-gec-areta
$ cd areta
$ pip install -r requirements.txt
```


## How to Reproduce

Scripts to run pipeline from finetuning step to evaluation step are provided
`hybrid-llm-rule-based-synthetic-data-generation-arabic-gec/scripts/paper` 

note: `suffix q: experiments on qalb-15 and suffix qz: experiments on qalb-15 + ZAEBUC`


Note: you will need to change the paths in the scripts to the appropriate paths

You can download ready to use checkpoints and datasets from [here](https://drive.google.com/file/d/1Am1VwfX-XcwF3VU3USqVxVQFyQaIVGNf/view?usp=sharing) 

in the download you'll find

`paper_outputs/`: model outputs for checkpoints used in the paper
`finetuning_data/`: datasets used for finetuning in the paper including the synthetic data
`best_checkpoints`: 
  `checkpoint-3000q`: best checkpoint for finetuning on qalb-15 real data only
  `checkpoint-3000qz`: best checkpoint for finetuning on qalb-15+ZAEBUC real data only
  `checkpoint-17000q`: best checkpoint for finetuning on qalb-15 real data + our synthetic data
  `checkpoint-11500qz`: best checkpoint for finetuning on qalb-15+ZAEBUC real data + our synthetic data

you can find code to run to synthesize your own gec data using our recipe here gec/corruption

you can run `python -m gec.corruption -h`

example: `scripts/synthetic_data/shard_0.sh`




## Acknowledgements
Code for alignment method, evaluation, modified ARETA version, pre/post processing scripts and the morphological disambiguation step curtsey of `alhafni-etal-2023-advancements` [https://github.com/CAMeL-Lab/arabic-gec/tree/master](https://github.com/CAMeL-Lab/arabic-gec/tree/master):


# Citation

How to cite instructions will be available after paper publication date
