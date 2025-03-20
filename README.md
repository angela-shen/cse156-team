# CSE 156 final project

This repository contains code and experiments for a model that answers complex yes/no questions. By decomposing questions, extracting evidence from a Wikipedia-based corpus, and applying reasoning, the model determines whether an answer should be "yes" or "no." An example question is: 

> "Would it be common to find a penguin in Miami?"

## Overview

The goal of this project is to build a structured methodology that can handle intricate queries across diverse topics. Two primary approaches are explored:
- **Baseline Model (RoBERTa):** A straightforward approach that decomposes the question and gathers evidence using separate components.
- **GPT-2 Based Model:** Enhances the baseline by integrating text generation with classification, using a modified GPT-2 model and additional fully connected layers for binary decision-making.

## Real-world Impacts

Demonstrating the capability of models to answer nuanced questions accurately underscores the progress in AI. As models like ChatGPT and AI-powered search engines become integral to daily tasks, developing reliable and precise question-answering systems is critical for personal, professional, and educational use.

## Methods

### Baseline (RoBERTa)
- **Architecture:**  
  - The model decomposes the input question into parts.
  - It searches for related evidence from a curated dataset (sourced from Wikipedia).
  - A reasoning component then determines the final yes/no answer.
- **Pros and Cons:**  
  - *Pros:* Simplicity, ease of manipulation, and rapid prototyping.
  - *Cons:* Reliance on separate models for decomposition/evidence and classification can hinder cohesive reasoning.

### GPT-2 Based Model
- **Core Architecture:**  
  - Built on a pre-trained GPT-2 model (GPT2LMHeadModel) for text generation.
  - A special token `[SEP]` is introduced to separate the generated decomposition from the evidence.
- **Classification Layers:**  
  - **FC1:** Maps the GPT-2 output to a maximum sequence length with ReLU activation.
  - **FC2:** Reduces the representation to 128 dimensions with ReLU activation.
  - **FC3:** Outputs a single value with Sigmoid activation, yielding a probability for binary classification.
- **Training Strategy:**  
  - For text generation (decomposition and evidence prediction), CrossEntropyLoss is minimized while GPT-2 parameters are updated and the FC layers are frozen.
  - For the classification task, Binary Cross-Entropy Loss (BCELoss) is minimized. In this phase, GPT-2 parameters are frozen, and the FC layers are trained.

## Experiments

### Baseline Experiments
- **Setup:**  
  - Initial training was conducted with 3 epochs to gauge performance.
  - Extended experiments involved 10 epochs with fine-tuning using a linear warmup scheduler and gradient clipping.
- **Findings:**  
  - Training and validation accuracy peaked early and then fluctuated.
  - The overall accuracy remained relatively low, indicating room for improvement.

### GPT-2 Experiments
- **Outcomes:**  
  - The GPT-2 based model showed an approximate 5% improvement over the baseline.
  - Despite the improvement, the accuracy is still below the average on benchmark leaderboards.
- **Challenges:**  
  - The model struggled with generating the desired patterns, such as consistently including special tokens ([SEP], ?, |).
  - The limited size of the fine-tuning dataset (~2,000 questions) likely contributes to these challenges.

For a more detailed analysis, please refer to our [Overleaf report](https://www.overleaf.com/project/67be58ba899878f079243ece).

## Conclusion and Future Work

While the RoBERTa baseline offers an accessible starting point, significant improvements are needed. Future work will focus on:
- Experimenting with different learning rates and decoding sampling methods.
- Developing methods to reduce the dependency of the classification task on generated outputs, possibly by integrating additional features or exploring alternative architectures.

## Getting Started

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/angela-shen/cse156-team.git
cd cse156-team

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
