# Attention-Based Aspect Extraction

This repository is a fork of [paper authors' repository](https://github.com/ruidan/Unsupervised-Aspect-Extraction) with following code improvements:

* python 3 compliant
* Keras 2 compliant
* Keras backend independent

In addition there is an additional functionality:

* seed words
* no need to specify embedding dimension with external embedding usage model

## Dependencies

* keras>=2.0
* tensorflow-gpu>=1.4
* numpy>=1.13

This code also tested to work with CNTK and MXNet. With MXNet there were some issues with Keras internals, hope they will be improved in future versions.

## Data and Preprocessing

You can download the original datasets (of Restaurant and Beer domains) in [[Download]](https://drive.google.com/open?id=1qzbTiJ2IL5ATZYNMp2DRkHvbFYsnOVAQ).

For preprocessing, put the decompressed zip file in the main folder and run:
```bash
python preprocess.py
python word2vec.py
```
respectively in `code/`. The preprocessed files and trained word embeddings for each domain will be saved in a folder `preprocessed_data/`.

You can also find the pre-processed datasets and the pre-trained word embeddings in [[Download]](https://drive.google.com/open?id=1L4LRi3BWoCqJt5h45J2GIAW9eP_zjiNc).
The zip file should be decompressed and put in the main folder.

## Train

For training, run in `code/` folder:

```bash
python train.py \
--emb-name ../preprocessed_data/$domain/w2v_embedding \
--domain $domain \
--out-dir ../output
```
where:
* `$domain` (`restaurant` or `beer`) is the corresponding domain,
* `--emb-name` is the path to the pre-trained word embeddings, it could be just a name of a file, then it will be searched in `../preprocessed_data/$domain/`, otherwise it will be searched by absolute path;
* `--out-dir` is the path of the output directory.

You can find more arguments/hyper-parameters defined in [code/train.py] with default values used in our experiments.

After training, two output files will be saved in `../output/$domain/`:
* `aspect.log` contains extracted aspects with top 100 words for each of them.
* `model_param` contains the saved model weights.

## Evaluation

For evaluation, run in `code/` folder:

```bash
python evaluation.py \
--domain $domain \
--out-dir ../output
```

Note that you should keep the values of arguments for evaluation the same as those for training (except `--emb-name`, you don't need to specify it), as we need to first rebuild the network architecture and then load the saved model weights.

This will output a file `att_weights` that contains the attention weights on all test sentences in `../output/$domain/`.

To assign each test sentence a gold aspect label, you need to first manually map each inferred aspect to a gold aspect label according to its top words, and then uncomment the bottom part in evaluation.py (line 136-144) for evaluaton using F scores.

One example of trained model for the restaurant domain has been put in `pre_trained_model/restaurant/`, and the corresponding aspect mapping has been provided in [code/evaluation.py](code/evaluation.py) (at the bottom).

## Cite

If you use the code, please consider citing original paper:

```tex
@InProceedings{he-EtAl:2017:Long2,
  author    = {He, Ruidan  and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel},
  title     = {An Unsupervised Neural Attention Model for Aspect Extraction},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics}
}
```
