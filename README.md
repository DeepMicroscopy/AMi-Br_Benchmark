# Benchmarking Deep Learning and Vision Foundation Models for Atypical vs. Normal Mitosis Classification with Cross-Dataset Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2506.21444v1-b31b1b.svg)](https://arxiv.org/abs/2506.21444v1)

This repository contains the datasets and training and inference scripts for our paper:

> **Benchmarking Deep Learning and Vision Foundation Models for Atypical vs. Normal Mitosis Classification with Cross-Dataset Evaluation**  
> *Sweta Banerjee, Viktoria Weiss, Taryn A. Donovan, Rutger A. Fick, Thomas Conrad, Jonas Ammeling, Nils Porsche, Robert Klopfleisch, Christopher Kaltenecker, Katharina Breininger, Marc Aubreville, Christof A. Bertram*  
> [arXiv:2506.21444v1](https://arxiv.org/abs/2506.21444v1)

---

## Abstract:
```
Atypical mitosis marks a deviation in the cell division process that has been shown be an independent prognostic marker for tumor malignancy. However, atypical mitosis classification remains challenging due to low prevalence, at times subtle morphological differences from normal mitotic figures, low inter-rater agreement among pathologists, and class imbalance in datasets. Building on the Atypical Mitosis dataset for Breast Cancer (AMi-Br), this study presents a comprehensive benchmark comparing deep learning approaches for automated atypical mitotic figure (AMF) classification, including end-to-end trained deep learning models, foundation models with linear probing, and foundation models fine-tuned with low-rank adaptation (LoRA). For rigorous evaluation, we further introduce two new held-out AMF datasets - AtNorM-Br, a dataset of mitotic figures from the TCGA breast cancer cohort, and AtNorM-MD, a multi-domain dataset of mitotic figures from a subset of the MIDOG++ training set. We found average balanced accuracy values of up to 0.8135, 0.7788, and 0.7723 on the in-domain AMi-Br and the out-of-domain AtNorm-Br and AtNorM-MD datasets, respectively. Our work shows that atypical mitotic figure classification, while being a challenging problem, can be effectively addressed through the use of recent advances in transfer learning and model fine-tuning techniques. We make all code and data used in this paper available in this github repository.

```
---

## Contents

- `AtNorM-Br/`: Dataset (TCGA-Breast atypical and normal mitoses)
- `AtNorM-MD/`: Dataset (Multi-domain dataset from MIDOG++ featuring atypical and normal mitoses)
- `End-to-End_DL_Baselines/`: Fully trained baselines
- `Foundation_Models/`: Linear probing and Low Rank Adaptation of foundation models

---



