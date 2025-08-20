# Dementia-Prediction-Model

This project is focused on building a **Machine Learning model** to predict dementia occurrence using demographic and clinical data. It leverages data preprocessing and training techniques to generate meaningful predictions.

## Dataset

The model uses the **OASIS Longitudinal Demographics** dataset containing clinical and demographic data relevant to dementia analysis.

- Dataset file: `oasis_longitudinal_demographics.xlsx`

## Technologies Used

- Python 
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Project Structure

├── main.py                               # Main Python script for training and prediction  
├── oasis_longitudinal_demographics.xlsx  # Input dataset  
├── Predictions.xlsx                      # Output file with predicted results  
├── README.md                             # Project documentation

## Getting Started

Follow the steps below to get this project up and running on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/Taranpreet10451/Dementia-Prediction.git
cd Dementia-Prediction
```

### 2. Install Dependencies

It’s recommended to use a virtual environment.

If you have a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the required packages manually:

```bash
pip install pandas numpy scikit-learn
```

### 3. Run the Script

Make sure the dataset is available in the project directory. Then run:

```bash
python main.py
```

This will generate a `Predictions.xlsx` file with model outputs.

## Output Example

After training, the model will output:

- Accuracy metrics
- Predictions saved in `Predictions.xlsx`
- Insights into dementia prediction
