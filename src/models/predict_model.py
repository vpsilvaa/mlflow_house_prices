import mlflow
import pandas as pd

logged_model = 'runs:/14f4252e0b9747afbd54352f72da8e7c/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


data = pd.read_csv("data/processed/casas_X.csv")

# Predict on a Pandas DataFrame.
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('data/processed/precos.csv')