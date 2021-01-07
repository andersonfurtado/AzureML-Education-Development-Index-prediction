input_sample = pd.DataFrame({"Pregnancies": pd.Series([0.0], dtype="float64"), "Glucose": pd.Series([0.0], dtype="float64"), "BloodPressure": pd.Series([0.0], dtype="float64"), "SkinThickness": pd.Series([0.0], dtype="float64"), "Insulin": pd.Series([0.0], dtype="float64"), "BMI": pd.Series([0.0], dtype="float64"), "DiabetesPedigreeFunction": pd.Series([0.0], dtype="float64"), "Age": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})