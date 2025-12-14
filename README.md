# 2025 Digital Image Processing Final Project
Crowd Counting - Group 14

This repo contains several .ipynb files in which we ran our experiments (Definining models, Preparing dataset, training, testing)
You can run them locally or import them to Google Colab

## Content guide

### First_MCNN_Test
A fork of an implementation of Multi Column Neural Network for Crowd counting. Contains definition of the model, dataloader and other files for test and training, plus a small modification on the model in which we try to improve its acccuracy. Later experiments with the MCNN's will be based on this architecture, but in a single .ipynb file instead.

### p2pnet_test.ipynb
One of our attempts to implement a new technique (Point-to-point estimation) into the original MCNN architecture. Due to implementation difficulties, lack of time and our priorization of other more promising approaches, we stopped working on it