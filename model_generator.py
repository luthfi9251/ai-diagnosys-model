from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import joblib
import random
import os


class ModelGenerator:
    model = None

    def set_model(self, name_model, model):
        self.name_model = name_model
        if model == "decision_tree":
            self.model = DecisionTreeClassifier()

    def set_model_from_local(self, path_to_model):
        #try:
        
        path = os.path.join(os.getcwd(), path_to_model)
        self.name_model = os.path.basename(path)
        self.model = joblib.load(path)
        # except:
        #     print("Error when load model")
        #     return "Error when load model"

    def get_model_params(self):
        return self.model.get_params(deep=True)

    def set_model_params(self, params):
        self.model.set_params(**params)

    def set_dataset(self, path_to_dataset, target_label):
        full_path = os.path.join(os.getcwd(),path_to_dataset)
        self.dataset = pd.read_csv(full_path)
        self.target_label = target_label

    def prepare_dataset(self, test_size):
        self.test_size = test_size
        x = self.dataset.drop(self.target_label, axis=1)
        y = self.dataset[self.target_label]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=0
        )

    def train_model(self, test_size):
        self.prepare_dataset(test_size)
        self.model.fit(self.x_train, self.y_train)
        return self.get_information_model()

    def get_information_model(self):
        y_pred = self.model.predict(self.x_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        return {
            "model_name": self.name_model,
            "accuracy": self.accuracy,
            "recall": self.recall,
            "precision": self.precision,
            "test_size": self.test_size,
        }

    def find_best_params_model(self, params_test, test_size, scoring_base="accuracy"):
        self.prepare_dataset(test_size)
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=params_test,
            n_iter=100,
            cv=5,
            scoring=scoring_base,
        )
        random_search.fit(self.x_train, self.y_train)
        return {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "accuracy_on_test": random_search.best_estimator_.score(
                self.x_test, self.y_test
            ),
        }

    def save_model(self):
        path = "models/"
        self.filename =  self.name_model.replace(" ", "") + "-" + str(random.randint(100,1000)) + ".joblib"
        if self.model == None:
            return "Please set model first!"
        joblib.dump(self.model, path + self.filename)
        return{
            "path": path + self.filename,
            "file_name": self.filename,
            "model_info": self.get_information_model()
        }
    
    def get_model(self):
        return self.model
        
        



# test = ModelGenerator()

#test.set_model_from_local('models/decisiontree1-279.joblib')
# test.set_model("decisiontree2","decision_tree")
# test.set_dataset("datasets/heart.csv", "output")
# test.train_model(test_size=0.1)
# print(test.get_information_model())
# param_grid = {
#     "criterion": ["gini", "entropy"],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }
# res = test.find_best_params_model(params_test=param_grid, test_size=0.2)
# print(res)
