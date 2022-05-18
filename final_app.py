from flask import Flask, request
import joblib
import numpy
import sklearn
import catboost

MODEL_PATH1 = 'model1/model1.pkl'
SCALER_X_PATH1 = 'model1/scaler_x1.pkl'
SCALER_Y_PATH1 = 'model1/scaler_y1.pkl'

MODEL_PATH2 = 'model2/model2.pkl'
SCALER_X_PATH2 = 'model2/scaler_x2.pkl'
SCALER_Y_PATH2 = 'model2/scaler_y2.pkl'

app = Flask(__name__)


@app.route("/predict_price", methods=['GET'])
def predict():
    args = request.args
    area = args.get('area', default=-1, type=float)
    studio = args.get('studio', default=-1, type=int)
    renovation = args.get('renovation', default=-1, type=int)
    day_difference_int = args.get('day_difference_int', default=-1, type=int)
    rooms = args.get('rooms', default=-1, type=int)
    open_plan = args.get('open_plan', default=-1, type=int)
    floor = args.get('floor', default=-1, type=int)
    model_version = args.get('model_version', type=int)

    model1 = joblib.load(MODEL_PATH1)
    sc_x1 = joblib.load(SCALER_X_PATH1)
    sc_y1 = joblib.load(SCALER_Y_PATH1)

    model2 = joblib.load(MODEL_PATH2)
    sc_x2 = joblib.load(SCALER_X_PATH2)
    sc_y2 = joblib.load(SCALER_Y_PATH2)

    if model_version == 1:
        required_params = [area, studio, renovation, day_difference_int]

        if any([i == -1 for i in required_params]):
            return '500 Internal server error', 500

        x = numpy.array([area, studio, renovation, day_difference_int]).reshape(1,-1)
        x = sc_x1.transform(x)

        result1 = model1.predict(x)
        result1 = numpy.exp(sc_y1.inverse_transform(result1.reshape(1,-1)))

        return str(result1[0][0])

    if model_version == 2:
        required_params = [rooms, studio, floor, open_plan, renovation]

        if any([i == -1 for i in required_params]):
            return '500 Internal server error', 500

        x = numpy.array([rooms, studio, floor, open_plan, renovation]).reshape(1, -1)
        x = sc_x2.transform(x)

        result2 = model2.predict(x)
        result2 = numpy.exp(sc_y2.inverse_transform(result2.reshape(1, -1)))

        return str(result2[0][0])


if __name__ == '__main__':
    app.run(debug=True, port=5441, host='0.0.0.0')