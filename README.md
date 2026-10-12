# ML_MINIPROJECT
# Link Failure Prediction and Localization in Cloud-Scale Networks

**Course:** UE24CS352A — Machine Learning Mini-Project

## Project summary
This project demonstrates supervised learning for classifying optical network links as **healthy** or **potentially failing** using two aggregated features:
- `X1 = sum(actual amplifier gain - target amplifier gain)` across 19 spans
- `X2 = sum(target span loss - actual span loss)` across 19 spans

The three classifiers are Logistic Regression, Gaussian Discriminant Analysis (implemented with Linear Discriminant Analysis), and a linear-kernel Support Vector Machine.

## Important dataset note
The supplied reference paper describes 6,000 data points from live network telemetry, but the PDF supplied for this assignment does not contain the original raw dataset. To make the code runnable for a live demonstration, `main.py` generates a **synthetic demonstration dataset**. The generated model scores must not be presented as reproductions of the paper's reported 99% / 97% figures or as evidence of real-world network performance. Replace the generator with the authorized real dataset if your instructor provides it.

## Requirements
- Python 3.10 or later recommended
- pip

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
From the project root:
```bash
python main.py
```
Optional sample count:
```bash
python main.py --samples 6000
```






