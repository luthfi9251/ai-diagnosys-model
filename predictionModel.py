import pandas as pd
import time
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score


class PredictionModel:
    def __init__(self, model=None, dataset=None, target_label="", test_size=0.2):
        self.model = model
        self.target_label = target_label
        self.test_size = test_size
        self.df = pd.read_csv(dataset)
        self.column_list = list(self.df.columns)

    def find_best_model(self, model, model_params, scoring_based="accuracy"):
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=model_params,
            n_iter=100,
            cv=5,
            scoring=scoring_based,
        )
        random_search.fit(self.x_train, self.y_train)

        # Menampilkan parameter terbaik
        print("Parameter terbaik:", random_search.best_params_)
        # Menampilkan skor validasi silang terbaik
        print("Skor validasi silang terbaik:", random_search.best_score_)
        accuracy = random_search.best_estimator_.score(self.x_test, self.y_test)
        print("Akurasi pada dataset tes:", accuracy)

    def generate_train_test(self):
        x = self.df.drop(self.target_label, axis=1)
        y = self.df[self.target_label]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=0
        )

    def get_information_model(self):
        y_pred = self.model.predict(self.x_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        print(f"Akurasi: {self.accuracy:.2f}")
        print(f"Recall: {self.recall:.2f}")
        print(f"Precision: {self.precision:.2f}")
        return {
            "accuracy": self.accuracy,
            "recall": self.recall,
            "precision": self.precision,
        }

    def train_model(self, show_info=False):
        self.model.fit(self.x_train, self.y_train)
        if show_info:
            self.get_information_model()

    def predict_data(self, data):
        df_data_predict = pd.DataFrame(
            data=[data],
            columns=filter(lambda x: x != self.target_label, self.column_list),
        )
        confidence_val = self.model.predict_proba(df_data_predict)[0]
        return {
            "prediction": self.model.predict(df_data_predict)[0],
            "confidence": confidence_val,
        }
