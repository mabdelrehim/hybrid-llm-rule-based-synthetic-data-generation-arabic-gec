# Grammatical Error Correction Alignment:
The [M2 Scorer](https://github.com/nusnlp/m2scorer) has well known issues when it comes to generating M2 files. It tends to cluster multiple tokens together in a single edit and this leads to penalizing models even if they generate partially correct answers. The tool in this repo introduces a flexible alignment algorithm to align words between two parallel sentences, where the source sentence has spelling and grammatical errors and the target sentence has the corrections.

The algorithm handles all alignment types: insertions, deletions, replacements, splits, and merges. For instance, given the following two sentences:

Source:
```
خالد : اممممممممممممممممم اذا بتروحووون العصر الساعه ٢ اوكي ماعندي مانع لاتتأخرون و كلمو احمد هالمر ة لوسمحتم
```

Target:
```
خالد ، اذا بتروحون العصر الساعة 2 اوكيه ما عندي مانع بس لا تتأخرون وكلمو أحمد هذه المرة لو سمحتم .
```

The algorithm would generate the following alignment:

|Source|Target|
|--------------------|----------------------------|
|خالد | خالد |
| : |  ،|
| اممممممممممممممممم| |
|اذا | اذا |
|بتروحووون | بتروحون |
|العصر | العصر |
|الساعه | الساعة |
| ٢ | 2 |
| اوكي  | اوكيه |
|ماعندي | ما عندي |
|مانع | مانع |
| | بس |
|لاتتأخرون | لا تتأخرون |
| و كلمو | وكلمو |
|احمد | أحمد |
| هالمر ة | هذه المرة |
| لوسمحتم | لو سمحتم |
| | . |

## Generating Alignment:

```python
python aligner.py --src /path/to/src --tgt /path/to/tgt --output /path/to/output
```

To run the alignment on the sample files:

```python
python aligner.py --src sample/src.txt --tgt sample/tgt.txt --output sample/alignment.txt
```

## Generating M2 Files:

After generating the word-level alignment, we provide scripts to generate the M2 file that could be used with the M2 scorer for evaluation.

```python 
python create_m2_file.py --src /path/to/src --tgt /path/to/tgt 
                         --align /path/to/alignment --output /path/to/output
```

To run this script on the provided sample files:

```python
python create_m2_file.py --src sample/src.txt --tgt sample/tgt.txt 
                         --align sample/alignment.txt --output sample/edits.m2
```

Alignment method proposed in:

```bibtex
@inproceedings{alhafni-etal-2023-advancements,
    title = "Advancements in {A}rabic Grammatical Error Detection and Correction: An Empirical Investigation",
    author = "Alhafni, Bashar  and
      Inoue, Go  and
      Khairallah, Christian  and
      Habash, Nizar",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.396",
    pages = "6430--6448",
    abstract = "Grammatical error correction (GEC) is a well-explored problem in English with many existing models and datasets. However, research on GEC in morphologically rich languages has been limited due to challenges such as data scarcity and language complexity. In this paper, we present the first results on Arabic GEC using two newly developed Transformer-based pretrained sequence-to-sequence models. We also define the task of multi-class Arabic grammatical error detection (GED) and present the first results on multi-class Arabic GED. We show that using GED information as auxiliary input in GEC models improves GEC performance across three datasets spanning different genres. Moreover, we also investigate the use of contextual morphological preprocessing in aiding GEC systems. Our models achieve SOTA results on two Arabic GEC shared task datasets and establish a strong benchmark on a recently created dataset. We make our code, data, and pretrained models publicly available.",
}
```





