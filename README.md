# Tennis Match Outcome Predictor Using LSTM

This project predicts the outcome of ATP tennis matches using historical data from 2000 to 2024. The model is built using PyTorch and utilizes a Long Short-Term Memory (LSTM) network to predict the winner between two given players, along with the probability of the predicted winner's success.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model](#model)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Example Outputs](#example-outputs)
- [Future Improvements](#future-improvements)

## Project Overview
The goal of this project is to predict the outcome of a tennis match between two players based on their historical performances. Using ATP match data from 2000 to 2024, we trained a Long Short-Term Memory (LSTM) network to learn patterns in players' performance and make predictions for future matches.

## Technologies Used
- Python
- PyTorch
- scikit-learn
- pandas
- numpy

## Dataset
The dataset consists of ATP tennis matches from 2000 to 2024. It includes details about players, match statistics, surface types, tournament rounds, and more. 

We preprocess the data to:
- Fill missing values in numerical columns.
- One-hot encode categorical columns such as surface, round, and player hand.
- Normalize the numerical features using `StandardScaler`.

## Model
The model is a Long Short-Term Memory (LSTM) neural network built with PyTorch. LSTMs are effective for time series and sequential data, making them suitable for predicting match outcomes based on historical player performances.

The LSTM takes as input the features of the match (such as players' seeds, rank, aces, double faults, etc.) and outputs the probability of a particular player winning the match.

## How to Run

### Prerequisites
Ensure you have Python 3.x installed along with the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/tennis-match-outcome-predictor.git
    cd tennis-match-outcome-predictor
    ```

2. Install the required packages:
    ```bash
    pip install torch pandas scikit-learn numpy
    ```

3. Ensure you have the ATP match datasets from 2000 to 2024 in CSV format. The filenames should follow this pattern: `atp_matches_YYYY.csv` (e.g., `atp_matches_2020.csv`, `atp_matches_2021.csv`).

### Running the Model

1. Run the training and prediction script:
    ```bash
    python tennis_lstm_model.py
    ```

2. The script will load and preprocess the dataset, train the LSTM model, and output the training loss. After training, the script will print the accuracy of the model on the test set.

3. You can then use the `predict_match()` function to predict the outcome between two players.

## Usage
### Predicting the Outcome Between Two Players

To predict the outcome between two players, use the `predict_match()` function in the script.

Example:
```python
# Predict outcome between Novak Djokovic and Roger Federer
print(predict_match("Novak Djokovic", "Roger Federer", model, matches_df, scaler))
```

Output:
```csharp
The predicted winner is Novak Djokovic with a 58.43% win probability.
```

