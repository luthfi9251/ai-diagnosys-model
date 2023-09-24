from predictionModel import PredictionModel
from sklearn.tree import DecisionTreeClassifier


def predict(data):
    algorithm = DecisionTreeClassifier(
        criterion="entropy",
        random_state=1832,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=4,
    )
    model_generator = PredictionModel(
        model=algorithm, dataset="../datasets/heart.csv", target_label="output"
    )
    model_generator.generate_train_test()
    model_generator.train_model()
    return model_generator.predict_data(data)


result = predict([57, 0, 0, 140, 241, 0, 1, 123, 1, 0.2, 1, 0, 3])
print(result)
