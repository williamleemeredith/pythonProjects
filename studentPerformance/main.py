import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from statistics import mean


def main():
    data = pd.read_csv("data/student-mat.csv", sep=";")
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    predict = "G3"

    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    (x_train,
     x_test,
     y_train,
     y_test) = model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    print("Acc: ", acc)
    print("Co: ", linear.coef_)
    print("Intercept: ", linear.intercept_)
    print()

    predictions = linear.predict(x_test)
    variances = []

    for x in range(len(predictions)):
        predicted_y = round(predictions[x], 0)
        x_value = x_test[x]
        y_value = y_test[x]
        variance = predicted_y - y_value
        variances.append(variance)
        print(predicted_y, x_value, y_value, variance)

    print(mean(variances))


if __name__ == "__main__":
    main()
