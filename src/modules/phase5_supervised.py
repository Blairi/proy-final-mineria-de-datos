import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def crear_carpeta_outputs(output_dir=OUTPUT_DIR):
    """
    Crea la carpeta donde se guardan las graficas y resultados de la fase 5.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def definir_modelos():
    """
    Define la linea base y las iteraciones del Arbol de Decision.

    La linea base predice siempre la clase mayoritaria. Sirve para
    comparar si el arbol aprende mas que una regla trivial.
    """
    modelos = {
        "dummy_mayoritaria": DummyClassifier(
            strategy="most_frequent"
        ),
        "arbol_base": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "arbol_profundidad_limitada": DecisionTreeClassifier(
            max_depth=4,
            random_state=RANDOM_STATE
        ),
        "arbol_prepoda_balanceado": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
    }
    return modelos


def obtener_probabilidades(modelo, X_test, clases):
    """
    Obtiene las probabilidades del modelo en el mismo orden de clases usado
    para evaluar matrices y metricas.
    """
    if not hasattr(modelo, "predict_proba"):
        return None

    probabilidades = modelo.predict_proba(X_test)
    clases_modelo = list(modelo.classes_)

    if any(clase not in clases_modelo for clase in clases):
        return None

    indices = [clases_modelo.index(clase) for clase in clases]
    return probabilidades[:, indices]


def calcular_metricas(y_test, y_pred, y_score, clases):
    """
    Calcula metricas macro para evaluar todas las clases con el mismo peso.
    """
    if y_score is None:
        roc_auc_macro = None
    else:
        try:
            roc_auc_macro = roc_auc_score(
                y_test,
                y_score,
                labels=clases,
                multi_class="ovr",
                average="macro"
            )
        except ValueError:
            roc_auc_macro = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "roc_auc_ovr_macro": roc_auc_macro,
    }


def imprimir_matriz_confusion(nombre_modelo, matriz, clases):
    """
    Imprime la matriz de confusion multiclase.
    """
    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real {clase}" for clase in clases],
        columns=[f"Pred {clase}" for clase in clases]
    )

    print(f"\nMatriz de confusion: {nombre_modelo}")
    print(matriz_df.to_string())


def graficar_matriz_confusion(matriz, clases, nombre_modelo, output_dir):
    """
    Genera y guarda la matriz de confusion del modelo evaluado.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clases,
        yticklabels=clases,
        ax=ax
    )
    ax.set_title(f"Matriz de confusion - {nombre_modelo}")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    plt.tight_layout()

    ruta = os.path.join(output_dir, f"fase5_matriz_confusion_{nombre_modelo}.png")
    plt.savefig(ruta, dpi=150)
    plt.show()
    plt.close(fig)
    return ruta


def entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test, output_dir):
    """
    Entrena los modelos, calcula metricas macro y guarda matrices de confusion.
    """
    modelos = definir_modelos()
    clases = sorted(y_train.unique())
    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_score = obtener_probabilidades(modelo, X_test, clases)
        metricas = calcular_metricas(y_test, y_pred, y_score, clases)

        matriz = confusion_matrix(y_test, y_pred, labels=clases)
        imprimir_matriz_confusion(nombre, matriz, clases)
        ruta_matriz = graficar_matriz_confusion(
            matriz, clases, nombre, output_dir
        )

        resultados.append({
            "modelo": nombre,
            **metricas,
            "ruta_matriz_confusion": ruta_matriz,
        })
        modelos_entrenados[nombre] = modelo

    tabla_resultados = pd.DataFrame(resultados)
    tabla_resultados = tabla_resultados.sort_values(
        by=["f1_macro", "recall_macro", "precision_macro", "roc_auc_ovr_macro"],
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    ruta_csv = os.path.join(output_dir, "fase5_resultados_supervisado.csv")
    tabla_resultados.to_csv(ruta_csv, index=False)

    print("\nComparacion de modelos supervisados:")
    columnas = [
        "modelo",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
    ]
    print(tabla_resultados[columnas].round(4).to_string(index=False))
    print(f"\nResultados guardados en: {ruta_csv}")

    return modelos_entrenados, tabla_resultados, clases


def seleccionar_mejor_modelo(modelos_entrenados, tabla_resultados):
    """
    Selecciona el modelo con mejor F1 macro.

    Debido al desbalance de clases, no se elige solamente por accuracy.
    En caso de empate, el ordenamiento prioriza recall macro y precision macro.
    """
    nombre_mejor = tabla_resultados.loc[0, "modelo"]
    modelo_mejor = modelos_entrenados[nombre_mejor]

    print(f"\nMejor modelo por F1 macro: {nombre_mejor}")
    print("Decision tecnica: se prioriza F1 macro por el desbalance de clases.")

    return nombre_mejor, modelo_mejor


def graficar_roc_multiclase(modelo, X_test, y_test, clases, nombre_modelo, output_dir):
    """
    Genera curva ROC multiclase con enfoque One-vs-Rest.
    """
    y_score = obtener_probabilidades(modelo, X_test, clases)
    if y_score is None:
        print("El modelo seleccionado no permite generar curva ROC.")
        return None

    y_test_bin = label_binarize(y_test, classes=clases)
    if y_test_bin.shape[1] < 2:
        print("La curva ROC multiclase requiere al menos dos clases.")
        return None

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, clase in enumerate(clases):
        positivos = y_test_bin[:, i].sum()
        negativos = len(y_test_bin) - positivos

        if positivos == 0 or negativos == 0:
            print(f"No se genera ROC para {clase}: faltan positivos o negativos.")
            continue

        fpr[clase], tpr[clase], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[clase] = auc(fpr[clase], tpr[clase])

    if not fpr:
        print("No fue posible generar curvas ROC con las clases disponibles.")
        return None

    puntos_fpr = np.unique(np.concatenate([fpr[clase] for clase in fpr]))
    tpr_macro = np.zeros_like(puntos_fpr)

    for clase in fpr:
        tpr_macro += np.interp(puntos_fpr, fpr[clase], tpr[clase])

    tpr_macro /= len(fpr)
    auc_macro = auc(puntos_fpr, tpr_macro)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        puntos_fpr,
        tpr_macro,
        label=f"Promedio macro (AUC = {auc_macro:.3f})",
        color="black",
        linewidth=2
    )

    for clase in fpr:
        ax.plot(
            fpr[clase],
            tpr[clase],
            linewidth=1.5,
            label=f"{clase} vs resto (AUC = {roc_auc[clase]:.3f})"
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"Curva ROC multiclase One-vs-Rest - {nombre_modelo}")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    ruta = os.path.join(output_dir, "fase5_roc_multiclase_ovr.png")
    plt.savefig(ruta, dpi=150)
    plt.show()
    plt.close(fig)

    print(f"Curva ROC multiclase guardada en: {ruta}")
    return ruta


def graficar_importancia_variables(modelo, columnas, nombre_modelo, output_dir):
    """
    Guarda la importancia de variables del arbol seleccionado.
    """
    if not hasattr(modelo, "feature_importances_"):
        print("El modelo seleccionado no tiene importancia de variables.")
        return None

    importancias = pd.Series(modelo.feature_importances_, index=columnas)
    importancias = importancias.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.plot(kind="barh", color="#4C78A8", ax=ax)
    ax.set_title(f"Importancia de variables - {nombre_modelo}")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Atributo")
    plt.tight_layout()

    ruta = os.path.join(output_dir, "fase5_importancia_variables_arbol_decision.png")
    plt.savefig(ruta, dpi=150)
    plt.show()
    plt.close(fig)

    print(f"Importancia de variables guardada en: {ruta}")
    print("\nImportancia de variables del modelo seleccionado:")
    print(importancias.sort_values(ascending=False).round(4).to_string())

    return ruta


def ejecutar(X_train, X_test, y_train, y_test):
    """
    Punto de entrada de la fase 5 supervisada.
    """
    print("\n==== FASE 5: MODELO SUPERVISADO ====")
    print("Problema: clasificacion multiclase de tenencia.")
    print("Algoritmo elegido: Arbol de Decision por su interpretabilidad.")
    print("Criterio de seleccion: F1 macro por desbalance de clases.")

    output_dir = crear_carpeta_outputs()

    print("\nDistribucion de clases en entrenamiento:")
    print(y_train.value_counts())
    print("\nDistribucion de clases en prueba:")
    print(y_test.value_counts())

    modelos_entrenados, tabla_resultados, clases = entrenar_y_evaluar_modelos(
        X_train, X_test, y_train, y_test, output_dir
    )

    nombre_mejor, modelo_mejor = seleccionar_mejor_modelo(
        modelos_entrenados, tabla_resultados
    )

    ruta_roc = graficar_roc_multiclase(
        modelo_mejor, X_test, y_test, clases, nombre_mejor, output_dir
    )
    ruta_importancias = graficar_importancia_variables(
        modelo_mejor, X_train.columns, nombre_mejor, output_dir
    )

    print("\nFase 5 completada. Graficas guardadas en carpeta outputs.")

    return {
        "modelo": modelo_mejor,
        "nombre_modelo": nombre_mejor,
        "resultados": tabla_resultados,
        "ruta_roc": ruta_roc,
        "ruta_importancias": ruta_importancias,
    }
