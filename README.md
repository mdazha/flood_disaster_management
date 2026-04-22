# Flood Disaster Management

This project is a research-oriented prototype for post-flood disaster response using image classification and priority-based ranking. It combines a convolutional neural network with a sorting-based decision step so that flood-affected locations can be ranked for relief delivery based on both flood severity and travel distance.

The work is based on the paper:
`An integrated convolutional neural network and sorting algorithm for image classification for efficient flood disaster management`  
Decision Analytics Journal, 2023.

## Project Goal

After a flood, relief teams often need to decide which locations should receive aid first. This project uses flood images to:

1. Classify the severity of flooding in each image.
2. Convert those model outputs into priority values.
3. Combine severity with distance information.
4. Sort locations so a drone-assisted delivery workflow can serve higher-priority locations first.

In the paper, the overall idea is to support autonomous or semi-autonomous drone operations in areas where access is difficult, communication may be limited, and rapid prioritization matters.

## How the Pipeline Works

The current script in this repository, [Priority_flood_disaster_management.py](/Users/mdazharu/flood_disaster_management/Priority_flood_disaster_management.py:1), follows this workflow:

1. Load a flood image dataset organized into `train`, `validation`, and `test` folders.
2. Fine-tune a pretrained CNN model using transfer learning.
3. Evaluate the model on test images and print classification metrics.
4. Generate per-image predictions and softmax probabilities.
5. Combine the prediction output with distance values to produce a delivery priority score.
6. Sort the resulting scores to rank locations from lower to higher or higher to lower priority depending on the reporting step.
7. Save a trained checkpoint and generate a confusion matrix figure.

## Models Used

The project supports two pretrained CNN backbones from `torchvision`:

- `Inception v3`
- `DenseNet-121`

The script is currently configured to use `Inception v3` by default. In the paper, Inception v3 performed slightly better than DenseNet on the authors' flood dataset:

- Inception v3: 83% accuracy
- DenseNet: 81% accuracy

The paper also reports macro F1, precision, and recall, showing a small but consistent advantage for Inception v3.

## Flood Severity Framing

The dataset contains three flood categories:

- `Minor_flooding`
- `Moderate_flooding`
- `Major_flooding`

In the research framing, more severe flooding should receive higher priority in the ranking stage. The paper describes the ranking idea as combining flood severity with distance so that nearby severe cases can be prioritized efficiently.

Conceptually:

`priority score = severity weight / distance`

Those scores are then sorted to determine an order for relief delivery.

## Dataset Structure

The script expects the dataset to be laid out like this:

```text
flood_disaster_management/
├── train/
│   ├── Minor_flooding/
│   ├── Moderate_flooding/
│   └── Major_flooding/
├── validation/
│   ├── Minor_flooding/
│   ├── Moderate_flooding/
│   └── Major_flooding/
└── test/
    ├── Minor_flooding/
    ├── Moderate_flooding/
    └── Major_flooding/
```

According to the paper, the original dataset was created by web scraping flood images and organizing them into those three classes. The reported dataset split in the paper was:

- Train: 250 minor, 290 moderate, 245 major
- Validation: 20 minor, 25 moderate, 20 major
- Test: 54 minor, 54 moderate, 53 major

## What the Script Produces

When you run the script, it performs training, evaluation, and ranking in one file. The main outputs are:

- Console logs for training and validation loss/accuracy
- Test-set evaluation metrics: accuracy, F1 score, precision, recall
- Printed probability vectors for test images
- Ranked image paths based on the computed priority values
- A saved model checkpoint: `dense_0.pth`
- A saved confusion matrix image: `confusematrx.png`

## Requirements

The script uses the following Python packages:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `seaborn`
- `scikit-learn`

If you run the script in Google Colab, it can also mount Google Drive. Outside Colab, that step is skipped safely in the current version.

## How to Run

From the project directory:

```bash
python3 Priority_flood_disaster_management.py
```

Before running, make sure:

- your dataset folders exist under the project root
- the class names match the folder names expected by the script
- the `test` directory contains the images you want ranked

## Research Scope and Practical Scope

This repository is best understood as a research prototype rather than a complete drone deployment system. It demonstrates the decision pipeline:

- classify flood severity from images
- incorporate distance into prioritization
- sort candidate delivery points for relief planning

It does not include full drone navigation, live telemetry, route optimization, or real-time field integration. Those would be natural next steps for turning the research idea into an operational disaster-response platform.

## Reference

Islam, M. A., Rashid, S. I., Hossain, N. U. I., Fleming, R., and Sokolov, A.  
`An integrated convolutional neural network and sorting algorithm for image classification for efficient flood disaster management`  
Decision Analytics Journal, Volume 7, 2023, Article 100225.
