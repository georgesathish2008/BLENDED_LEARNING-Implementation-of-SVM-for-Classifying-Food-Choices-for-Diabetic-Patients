# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Support Vector Machine (SVM) is used to classify data into different classes based on selected
features.
2. StandardScaler is applied to normalize the feature values for better model performance.
3. GridSearchCV is used to find the best hyperparameters for the SVM model.
4. Model Evaluation is performed using accuracy score, classification report, and confusion matrix.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: SATHISH H
RegisterNumber: 212225240142
*/
#Step 1 import libraries
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
#Step 2
#Load dataset
data=pd.read_csv('food_items_binary.csv.xls')
print(data.head())
print(data.columns)
#selecting features and target
features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target='class'
x=data[features]
y=data[target]
#step 4
#splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)
#step 5 feature scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#step 6 model training
svm = SVC()
#Setup hyperparameter grid for tuning
param_grid={
'C':[0.1,1,10,100], #regularization parameter
'kernel':['linear','rbf'], #kernel types
'gamma':['scale','auto'] #kernel coefficient
}
#initialize gridsearchCV
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(x_train,y_train)
#Extract best model
best_model=grid_search.best_estimator_
print("Name: Hari Prasath M")
print("Register No:212225100015")
print("Best Parameters:", grid_search.best_params_)
#step 7 Model Evaluation
y_pred=best_model.predict(x_test)
#calculate accuracy and print classification matrix
accuracy=accuracy_score(y_test,y_pred)
print("Name: SATHISH H")
print("Register No:212225240142")
print("Accuracy:",accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<Figure size 640x480 with 2 Axes><img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/3c94261c-def8-4ba4-a888-f62902ae483d" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
