# Benchmarking Deep Learning and Vision Foundation Models for Atypical vs. Normal Mitosis Classification with Cross-Dataset Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2506.21444v1-b31b1b.svg)](https://arxiv.org/abs/2506.21444v1)

This repository contains the datasets and training and inference scripts for our paper:

> **Benchmarking Deep Learning and Vision Foundation Models for Atypical vs. Normal Mitosis Classification with Cross-Dataset Evaluation**  
> *Sweta Banerjee, Viktoria Weiss, Taryn A. Donovan, Rutger A. Fick, Thomas Conrad, Jonas Ammeling, Nils Porsche, Robert Klopfleisch, Christopher Kaltenecker, Katharina Breininger, Marc Aubreville, Christof A. Bertram*  
> [arXiv:2506.21444v1](https://arxiv.org/abs/2506.21444v1)

---

## Overview

#### Atypical mitoses mark a deviation in the cell division process that can be an independent prognostically relevant marker for tumor malignancy. However, their identification remains challenging due to low prevalence, at times subtle morphological differences from normal mitoses, low inter-rater agreement among pathologists, and class imbalance in datasets. Building on the Atypical Mitosis dataset for Breast Cancer (AMi-Br), this study presents a comprehensive benchmark comparing deep learning approaches for automated atypical mitotic figure (AMF) classification, including baseline models, foundation models with linear probing, and foundation models fine-tuned with low-rank adaptation (LoRA). For rigorous evaluation, we further introduce two new hold-out AMF datasets - AtNorM-Br, a dataset of mitoses from the The TCGA breast cancer cohort, and AtNorM-MD, a multi-domain dataset of mitoses from the MIDOG++ training set. We found average balanced accuracy values of up to 0.8135, 0.7696, and 0.7705 on the in-domain AMi-Br and the out-of-domain AtNorm-Br and AtNorM-MD datasets, respectively, with the results being particularly good for LoRA-based adaptation of the Virchow-line of foundation models. Our work shows that atypical mitosis classification, while being a challenging problem, can be effectively addressed through the use of recent advances in transfer learning and model fine-tuning techniques. We make available all code and data used in this paper in this github repository.
---

## 📁 Contents

- `AtNorM-Br/`: Dataset (TCGA-Breast atypical and normal mitoses)
- `AtNorM-MD/`: Dataset (Multi-domain dataset from MIDOG++ featuring atypical and normal mitoses)
- `End-to-End_DL_Baselines/`: Fully trained baselines
- `Foundation_Models/`: Linear probing and Low Rand Adaptation of foundation models

---



