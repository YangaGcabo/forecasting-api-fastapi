import joblib

def load_model(path="models/model.pkl"):
    """
    Load trained forecasting model.
    """
    return joblib.load(path)
