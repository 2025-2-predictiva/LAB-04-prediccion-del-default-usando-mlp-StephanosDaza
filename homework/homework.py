import os
import gzip
import pickle
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

def clean_dataset(df):
    """Paso 1: Limpieza de datos"""
    df = df.copy()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    return df

def load_data():
    """Carga los datasets desde los zips"""
    train_df = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip")
    return train_df, test_df

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calcula metricas y devuelve los diccionarios"""
    metrics = {
        'type': 'metrics',
        'dataset': dataset_name,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    cm = confusion_matrix(y_true, y_pred)
    cm_dict = {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {"predicted_0": int(cm[0,0]), "predicted_1": int(cm[0,1])},
        'true_1': {"predicted_0": int(cm[1,0]), "predicted_1": int(cm[1,1])}
    }
    return metrics, cm_dict

def main():
    # --- Paso 1 y 2: Carga y Limpieza de Datos ---
    train_df, test_df = load_data()
    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    x_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    x_test = test_df.drop(columns=['default'])
    y_test = test_df['default']

    # --- Paso 3: Construccion del Pipeline ---
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = [col for col in x_train.columns if col not in cat_cols]

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('std', StandardScaler(), num_cols) 
        ],
        remainder='passthrough'
    )

    # Pipeline con los pasos requeridos
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_classif)),
        ('pca', PCA()), 
        ('classifier', MLPClassifier(max_iter=15000, random_state=21)) 
    ])

    # --- Paso 4: Optimizacion de Hiperparametros ---
    param_grid = {
        'selector__k': [20],
        'pca__n_components': [None],
        'classifier__hidden_layer_sizes': [(50, 30, 40, 60)], 
        'classifier__alpha': [0.26],                          
        'classifier__learning_rate_init': [0.001]
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    model.fit(x_train, y_train)

    # --- Paso 5: Guardar el modelo ---
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

    # --- Paso 6 y 7: Calcular metricas y Guardar ---
    os.makedirs("files/output", exist_ok=True)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular diccionarios
    train_metrics, train_cm = calculate_metrics(y_train, y_train_pred, 'train')
    test_metrics, test_cm = calculate_metrics(y_test, y_test_pred, 'test')

    # Guardar Correctamente
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')
        f.write(json.dumps(train_cm) + '\n')
        f.write(json.dumps(test_cm) + '\n')

    print("Tarea completada exitosamente.")

if __name__ == "__main__":
    main()