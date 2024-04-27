import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



path_data_set='pima_indians_diabetes.txt'
df=pd.read_csv(path_data_set)
df.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']

X = np.array(df.drop(['Class'], axis = 1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size =0.2, random_state = 42)


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42) 
model.fit(X_train, y_train) 
y_pred1=model.predict(X_test) 
