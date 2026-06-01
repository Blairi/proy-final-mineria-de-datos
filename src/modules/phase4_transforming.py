import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def separar_features_etiqueta(df):
    """
    Separa el dataframe en X (atributos) e y (etiqueta).
    """
    X = df.drop(columns=['tenencia_cat'])
    y = df['tenencia_cat']
    return X, y


def particionar(X, y, test_size=0.2, random_state=42):
    """
    Divide en entrenamiento y prueba manteniendo la proporción
    de clases con stratify, para evitar subrepresentación de
    clases minoritarias (Prestada) en alguno de los conjuntos.
    """
    print(f"\n-- Particionando dataset (train {int((1-test_size)*100)}% / test {int(test_size*100)}%) --")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y             # respeta proporción de clases
    )

    print(f"Total registros : {len(X):,}")
    print(f"Entrenamiento   : {len(X_train):,}")
    print(f"Prueba          : {len(X_test):,}")

    print("\nDistribución de clases en entrenamiento:")
    print(y_train.value_counts(normalize=True).round(3))
    print("\nDistribución de clases en prueba:")
    print(y_test.value_counts(normalize=True).round(3))

    return X_train, X_test, y_train, y_test


def escalar(X_train, X_test):
    """
    Normaliza con StandardScaler.
    - fit_transform sobre train: aprende media y desviación
    - transform sobre test: aplica la misma escala aprendida

    Razón: el scaler NUNCA debe ver datos de prueba durante
    el ajuste — hacerlo introduciría data leakage y haría
    que las métricas finales no reflejen rendimiento real.

    Se usa para el algoritmo no supervisado (K-Means).
    El árbol de decisión no requiere escalado pero se
    generan ambas versiones para tenerlas disponibles.
    """
    print("\n-- Escalando atributos (StandardScaler) --")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns, index=X_train.index)
    X_test_sc  = pd.DataFrame(X_test_sc,  columns=X_test.columns,  index=X_test.index)

    print("Escalado completado.")
    print(f"\nMedia por atributo (train escalado):\n{X_train_sc.mean().round(4)}")

    return X_train_sc, X_test_sc, scaler


def validar_resultado(X_train, X_test, y_train, y_test):
    print("\n-- Validacion final fase 4 --")
    print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")


def ejecutar(df_work):
    print("==== FASE 4: TRANSFORMACIÓN ====")

    X, y = separar_features_etiqueta(df_work)
    X_train, X_test, y_train, y_test = particionar(X, y)
    X_train_sc, X_test_sc, scaler    = escalar(X_train, X_test)
    validar_resultado(X_train, X_test, y_train, y_test)

    print("\nTransformación completada. Dataset listo para módulos 5 y 6.")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler
