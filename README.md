# Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

A state-of-the-art, real-time credit card fraud detection system that addresses the critical business problem of detecting fraudulent transactions with high recall while minimizing false positives. This project demonstrates advanced machine learning techniques for handling highly imbalanced datasets and provides explainable AI insights through SHAP values.

## Key Features

- **Advanced Modeling**: Utilizes XGBoost with SMOTE for robust fraud detection on highly imbalanced data
- **Explainability**: Integrates SHAP values to explain model predictions, providing transparency and trust
- **Real-time Deployment**: Features a low-latency FastAPI backend for inference (<100ms response time)
- **Interactive Dashboard**: Includes a Streamlit dashboard for visualizing model performance and testing transactions
- **Production-Ready**: Complete with API documentation, unit tests, and deployment configurations

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.97 |
| Precision | 0.94 |
| Recall | 0.91 |
| F1-Score | 0.92 |

## Dataset

This project uses the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle.

**Dataset Details:**
- 284,807 transactions over 2 days (September 2013)
- 492 fraudulent transactions (0.172% of total)
- Features are PCA-transformed for privacy
- Highly imbalanced dataset requiring specialized techniques

**Citation:**
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
```

## Directory Structure

```
credit_card_fraud/
├── data/                      # Dataset (tracked with Git LFS)
│   └── creditcard.csv
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_modeling.ipynb     # Model training and evaluation
│   └── 03_explainability.ipynb # SHAP analysis
├── src/                       # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── deployment/                # Deployment code
│   ├── api.py                # FastAPI backend
│   └── streamlit_app.py      # Streamlit dashboard
├── models/                    # Trained models
│   └── xgb_fraud_model.pkl
├── results/                   # Model outputs and metrics
├── tests/                     # Unit tests
├── social_media/             # Graphics for promotion
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Git
- Git LFS (for large file storage)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install Git LFS** (if not already installed)
   ```bash
   # Windows (using Chocolatey)
   choco install git-lfs

   # macOS (using Homebrew)
   brew install git-lfs

   # Linux (Debian/Ubuntu)
   sudo apt-get install git-lfs

   # Initialize Git LFS
   git lfs install
   ```

3. **Pull LFS files**
   ```bash
   git lfs pull
   ```

4. **Create a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit Dashboard

```bash
streamlit run deployment/streamlit_app.py
```

The dashboard provides:
- Model performance metrics visualization
- Real-time transaction fraud prediction
- SHAP explainability visualizations
- Interactive feature importance charts

### Running the FastAPI Backend

```bash
cd deployment
uvicorn api:app --reload
```

API will be available at `http://localhost:8000`

API documentation (Swagger UI): `http://localhost:8000/docs`

### Running Tests

```bash
pytest tests/
```

## API Endpoints

### POST /predict
Predict fraud probability for a transaction

**Request Body:**
```json
{
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  ...
  "V28": -0.021053,
  "Amount": 149.62
}
```

**Response:**
```json
{
  "fraud_probability": 0.0342,
  "prediction": "Not Fraud",
  "shap_values": {...}
}
```

## Model Development

### Data Preprocessing
- Handled highly imbalanced dataset (0.172% fraud rate)
- Applied SMOTE (Synthetic Minority Over-sampling Technique)
- Scaled amount feature using StandardScaler
- Train-test split with stratification

### Model Training
- **Algorithm**: XGBoost Classifier
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Evaluation Metrics**: ROC-AUC, Precision, Recall, F1-Score
- **Class Balancing**: SMOTE + class weights

### Explainability
- SHAP (SHapley Additive exPlanations) for feature importance
- Individual prediction explanations
- Global feature impact analysis

## Deployment

### Streamlit Cloud
```bash
# Ensure .streamlit/config.toml is configured
streamlit run deployment/streamlit_app.py
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Technologies Used

- **Python**: Core programming language
- **XGBoost**: Gradient boosting framework
- **SHAP**: Explainable AI library
- **FastAPI**: High-performance web framework
- **Streamlit**: Interactive dashboard framework
- **Scikit-learn**: Machine learning utilities
- **Pandas & NumPy**: Data manipulation
- **Plotly & Seaborn**: Data visualization
- **imbalanced-learn**: SMOTE implementation

## Future Enhancements

- [ ] Implement real-time monitoring dashboard
- [ ] Add AutoML for hyperparameter optimization
- [ ] Deploy to cloud platforms (AWS/GCP/Azure)
- [ ] Add model versioning with MLflow
- [ ] Implement A/B testing framework
- [ ] Add anomaly detection with autoencoders
- [ ] Create mobile-responsive UI

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Victor Collins Oppon**

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Machine Learning Group - Universite Libre de Bruxelles
- SHAP library by Scott Lundberg

## Contact

For questions or collaboration opportunities, please reach out via GitHub issues.

---

**Note**: This project is for educational and portfolio purposes. It demonstrates advanced machine learning techniques for fraud detection but should not be used in production without proper testing, validation, and security measures.
