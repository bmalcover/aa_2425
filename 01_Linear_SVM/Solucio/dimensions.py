import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.svm import SVC

# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train) # Obtenim els paràmetres del test
X_test_transformed = scaler.transform(X_test)

#Entrenam una SVM linear (classe SVC)
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)

print(f'Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')