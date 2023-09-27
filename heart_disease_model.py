from model_generator import ModelGenerator
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os


class HeartDiseaseModel:
    default_dataset = "datasets/heart.csv"
    target_label= "output"
    params_temp = {}

    def __init__(self):
        self.model_builder = ModelGenerator()
        self.dataset_handler()

    def dataset_handler(self):
        self.df = pd.read_csv(os.path.join(os.getcwd(),self.default_dataset))
        self.column_list = list(self.df.columns)
    
    def predict_use_local_model(self, path_to_model, data_to_predict):
        model_builder = self.model_builder
        model_builder.set_model_from_local(path_to_model)
        model = model_builder.get_model()
        df_data_predict = pd.DataFrame(
            data=[data_to_predict],
            columns=filter(lambda x: x != self.target_label, self.column_list),
        )
        confidence_val = model.predict_proba(df_data_predict)[0]
        return {
            "prediction": model.predict(df_data_predict)[0],
            "confidence": confidence_val,
        }
    
    def generate_new_model(self,name, params=None):
        if(params == None):
            params = {'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': None, 'criterion': 'entropy'}
        self.params_temp = params
        model_builder = self.model_builder
        model_builder.set_model(name,"decision_tree")
        model_builder.set_model_params(params)
        model_builder.set_dataset(self.default_dataset,self.target_label)
        return model_builder.train_model(test_size=0.1)
    
    def save_model_to_local(self, name, params=None):
        self.generate_new_model(name,params)
        return self.model_builder.save_model()
        


    

result = HeartDiseaseModel().predict_use_local_model("models/model112-898.joblib",[42,1,3,145,130,1,0,120,1,1.0,2,0,0])
#result = HeartDiseaseModel().save_model_to_local("model112")
print(result)
