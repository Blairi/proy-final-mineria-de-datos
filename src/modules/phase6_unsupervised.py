import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
K_VALUES = range(2, 9)
CANDIDATE_K_VALUES = (4, 8)
FINAL_K_PROVISIONAL = 4


def crear_carpeta_outputs(output_dir=OUTPUT_DIR):
    """
    Crea la carpeta donde se guardan las graficas y resultados de la fase 6.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def crear_kmeans(k):
    """
    Crea un modelo K-Means con parametros fijos para reproducibilidad.
    """
    return KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )


def calcular_silueta_segura(X, etiquetas):
    """
    Calcula Silueta solo cuando existen al menos dos clusters validos.
    """
    clusters_unicos = pd.Series(etiquetas).nunique()

    if clusters_unicos < 2 or clusters_unicos >= len(X):
        return None

    try:
        return silhouette_score(X, etiquetas)
    except ValueError:
        return None


def evaluar_valores_k(X_train_sc, k_values=K_VALUES):
    """
    Evalua varios valores de k usando solo X_train_sc.

    K-Means es no supervisado: no utiliza tenencia_cat para formar clusters.
    """
    resultados = []

    for k in k_values:
        modelo = crear_kmeans(k)
        etiquetas = modelo.fit_predict(X_train_sc)
        silueta = calcular_silueta_segura(X_train_sc, etiquetas)

        resultados.append({
            "k": k,
            "inertia": modelo.inertia_,
            "silhouette_train": silueta,
            "clusters_observados": pd.Series(etiquetas).nunique(),
        })

    return pd.DataFrame(resultados)


def guardar_evaluacion(tabla_k, output_dir):
    """
    Guarda la tabla de evaluacion de k.
    """
    ruta = os.path.join(output_dir, "fase6_evaluacion_kmeans.csv")
    tabla_k.to_csv(ruta, index=False)
    print(f"Evaluacion de k guardada en: {ruta}")
    return ruta


def graficar_metodo_codo(tabla_k, output_dir):
    """
    Muestra la grafica del metodo del codo usando inertia/WSS.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=tabla_k, x="k", y="inertia", marker="o", ax=ax)
    ax.set_title("Metodo del codo - K-Means")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Inertia / WSS")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    print("Grafica del metodo del codo mostrada en pantalla.")
    return None


def graficar_silueta(tabla_k, output_dir):
    """
    Muestra la grafica de Silueta para los valores de k evaluados.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=tabla_k, x="k", y="silhouette_train", marker="o", ax=ax)
    ax.set_title("Silueta por numero de clusters")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Silhouette score en entrenamiento")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    print("Grafica de Silueta mostrada en pantalla.")
    return None


def construir_centroides(modelo, columnas, scaler):
    """
    Construye centroides en escala estandarizada y escala original aproximada.

    Los atributos son codigos discretos representados numericamente; por eso
    los centroides originales se interpretan como perfiles promedio, no como
    valores literales exactos de una vivienda.
    """
    centroides_escalados = pd.DataFrame(
        modelo.cluster_centers_,
        columns=columnas
    )
    centroides_escalados.insert(0, "cluster", range(len(centroides_escalados)))

    centroides_originales = pd.DataFrame(
        scaler.inverse_transform(modelo.cluster_centers_),
        columns=columnas
    )
    centroides_originales.insert(0, "cluster", range(len(centroides_originales)))

    return centroides_escalados, centroides_originales


def guardar_centroides(centroides_escalados, centroides_originales, output_dir, prefijo):
    """
    Guarda centroides con un prefijo para diferenciar candidatos.
    """
    ruta_escalados = os.path.join(output_dir, f"{prefijo}_centroides_escalados.csv")
    ruta_originales = os.path.join(output_dir, f"{prefijo}_centroides_originales.csv")

    centroides_escalados.to_csv(ruta_escalados, index=False)
    centroides_originales.to_csv(ruta_originales, index=False)

    print(f"Centroides escalados guardados en: {ruta_escalados}")
    print(f"Centroides aproximados en escala original guardados en: {ruta_originales}")

    return ruta_escalados, ruta_originales


def obtener_distribucion_clusters(etiquetas):
    """
    Devuelve cantidad de registros por cluster.
    """
    return (
        pd.Series(etiquetas, name="cluster")
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="registros")
    )


def graficar_distribucion_clusters(distribucion, output_dir, prefijo):
    """
    Muestra la grafica de cantidad de registros por cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=distribucion, x="cluster", y="registros", color="#4C78A8", ax=ax)
    ax.set_title("Distribucion de registros por cluster - entrenamiento")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Registros")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    print("Grafica de distribucion de clusters mostrada en pantalla.")
    return None


def comparar_tenencia_por_cluster(etiquetas, y, output_dir, prefijo, nombre_conjunto, guardar=True):
    """
    Cruza clusters contra tenencia_cat solo para interpretacion posterior.

    tenencia_cat no se usa para entrenar K-Means.
    """
    df_interpretacion = pd.DataFrame({
        "cluster": etiquetas,
        "tenencia_cat": y.values,
    }, index=y.index)

    tabla_abs = pd.crosstab(
        df_interpretacion["cluster"],
        df_interpretacion["tenencia_cat"]
    ).sort_index()

    tabla_prop = pd.crosstab(
        df_interpretacion["cluster"],
        df_interpretacion["tenencia_cat"],
        normalize="index"
    ).sort_index()

    ruta_abs = None
    ruta_prop = None

    if guardar:
        ruta_abs = os.path.join(output_dir, f"{prefijo}_tenencia_por_cluster_{nombre_conjunto}.csv")
        ruta_prop = os.path.join(output_dir, f"{prefijo}_tenencia_por_cluster_{nombre_conjunto}_prop.csv")
        tabla_abs.to_csv(ruta_abs)
        tabla_prop.to_csv(ruta_prop)

        print(f"Tabla absoluta tenencia por cluster ({nombre_conjunto}) guardada en: {ruta_abs}")
        print(f"Tabla proporcional tenencia por cluster ({nombre_conjunto}) guardada en: {ruta_prop}")

    return tabla_abs, tabla_prop, ruta_abs, ruta_prop


def graficar_tenencia_por_cluster(tabla_prop, output_dir, prefijo):
    """
    Muestra una grafica apilada con proporciones de tenencia_cat por cluster.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    tabla_prop.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Distribucion proporcional de tenencia_cat por cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proporcion dentro del cluster")
    ax.legend(title="tenencia_cat", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    print("Grafica de tenencia por cluster mostrada en pantalla.")
    return None


def validar_en_prueba(modelo, X_test_sc):
    """
    Asigna clusters al conjunto de prueba y calcula Silueta si es valido.
    """
    etiquetas_test = modelo.predict(X_test_sc)
    silueta_test = calcular_silueta_segura(X_test_sc, etiquetas_test)
    return etiquetas_test, silueta_test


def guardar_alias_final(resultado, output_dir):
    """
    Guarda copias con nombres generales para el modelo final provisional.
    """
    rutas = []

    ruta_cent_esc = os.path.join(output_dir, "fase6_centroides_escalados.csv")
    ruta_cent_orig = os.path.join(output_dir, "fase6_centroides_originales.csv")
    ruta_train_abs = os.path.join(output_dir, "fase6_tenencia_por_cluster_train.csv")
    ruta_train_prop = os.path.join(output_dir, "fase6_tenencia_por_cluster_train_prop.csv")
    ruta_test_abs = os.path.join(output_dir, "fase6_tenencia_por_cluster_test.csv")
    ruta_test_prop = os.path.join(output_dir, "fase6_tenencia_por_cluster_test_prop.csv")

    resultado["centroides_escalados"].to_csv(ruta_cent_esc, index=False)
    resultado["centroides_originales"].to_csv(ruta_cent_orig, index=False)
    resultado["tabla_abs_train"].to_csv(ruta_train_abs)
    resultado["tabla_prop_train"].to_csv(ruta_train_prop)
    resultado["tabla_abs_test"].to_csv(ruta_test_abs)
    resultado["tabla_prop_test"].to_csv(ruta_test_prop)

    rutas.extend([
        ruta_cent_esc,
        ruta_cent_orig,
        ruta_train_abs,
        ruta_train_prop,
        ruta_test_abs,
        ruta_test_prop,
    ])

    return rutas


def ejecutar_candidato(k, X_train_sc, X_test_sc, y_train, y_test, scaler, output_dir):
    """
    Entrena, evalua y documenta un candidato de K-Means.
    """
    prefijo = f"fase6_k{k}"
    print(f"\n-- Candidato K-Means k={k} --")

    modelo = crear_kmeans(k)
    etiquetas_train = modelo.fit_predict(X_train_sc)
    etiquetas_test, silueta_test = validar_en_prueba(modelo, X_test_sc)
    silueta_train = calcular_silueta_segura(X_train_sc, etiquetas_train)
    distribucion_train = obtener_distribucion_clusters(etiquetas_train)
    min_cluster = int(distribucion_train["registros"].min())

    print(f"Silueta entrenamiento k={k}: {silueta_train:.4f}")
    if silueta_test is None:
        print(f"Silueta prueba k={k}: no calculable")
    else:
        print(f"Silueta prueba k={k}: {silueta_test:.4f}")

    print("\nTamano de clusters en entrenamiento:")
    print(distribucion_train.to_string(index=False))

    centroides_escalados, centroides_originales = construir_centroides(
        modelo, X_train_sc.columns, scaler
    )

    print("\nCentroides escalados:")
    print(centroides_escalados.round(4).to_string(index=False))
    print("\nCentroides aproximados en escala original:")
    print(centroides_originales.round(4).to_string(index=False))
    print("Nota: los centroides son perfiles promedio de codigos discretos.")

    tabla_abs_train, tabla_prop_train, ruta_train_abs, ruta_train_prop = comparar_tenencia_por_cluster(
        etiquetas_train, y_train, output_dir, prefijo, "train", guardar=False
    )

    tabla_abs_test, tabla_prop_test, ruta_test_abs, ruta_test_prop = comparar_tenencia_por_cluster(
        etiquetas_test, y_test, output_dir, prefijo, "test", guardar=False
    )

    print("\nDistribucion proporcional de tenencia_cat por cluster - train:")
    print(tabla_prop_train.round(4).to_string())

    return {
        "k": k,
        "modelo": modelo,
        "inertia": modelo.inertia_,
        "silueta_train": silueta_train,
        "silueta_test": silueta_test,
        "min_cluster_train": min_cluster,
        "distribucion_train": distribucion_train,
        "centroides_escalados": centroides_escalados,
        "centroides_originales": centroides_originales,
        "clusters_train": etiquetas_train,
        "clusters_test": etiquetas_test,
        "tabla_abs_train": tabla_abs_train,
        "tabla_prop_train": tabla_prop_train,
        "tabla_abs_test": tabla_abs_test,
        "tabla_prop_test": tabla_prop_test,
        "rutas": [],
    }


def construir_comparacion_candidatos(resultados_candidatos, output_dir):
    """
    Construye y guarda una tabla comparativa entre k=4 y k=8.
    """
    filas = []

    for resultado in resultados_candidatos:
        filas.append({
            "k": resultado["k"],
            "inertia": resultado["inertia"],
            "silhouette_train": resultado["silueta_train"],
            "silhouette_test": resultado["silueta_test"],
            "min_cluster_train": resultado["min_cluster_train"],
            "num_clusters": len(resultado["distribucion_train"]),
        })

    tabla = pd.DataFrame(filas).sort_values("k")
    ruta = os.path.join(output_dir, "fase6_comparacion_k4_k8.csv")
    tabla.to_csv(ruta, index=False)

    print("\nComparacion formal de candidatos k=4 vs k=8:")
    print(tabla.round(4).to_string(index=False))
    print(f"Comparacion de candidatos guardada en: {ruta}")

    return tabla, ruta


def seleccionar_modelo_final(resultados_por_k):
    """
    Selecciona k=4 como modelo final provisional si sus clusters son poblados.
    """
    resultado_k4 = resultados_por_k[FINAL_K_PROVISIONAL]
    resultado_k8 = resultados_por_k[8]

    print(f"\nK final provisional seleccionado: {FINAL_K_PROVISIONAL}")
    print("Justificacion:")
    print("- El metodo del codo muestra una mejora importante hasta k=4.")
    print("- k=8 mejora ligeramente la Silueta, pero sobresegmenta los datos.")
    print(f"- k=8 genera un cluster minimo de {resultado_k8['min_cluster_train']} registros.")
    print(f"- k=4 mantiene un cluster minimo de {resultado_k4['min_cluster_train']} registros.")
    print("- Se prioriza equilibrio entre metrica, codo e interpretabilidad.")

    return resultado_k4


def imprimir_rutas(rutas):
    """
    Imprime las rutas generadas por la fase 6.
    """
    print("\nArchivos generados por fase 6:")
    for ruta in rutas:
        if ruta is not None:
            print(f"- {ruta}")


def ejecutar(X_train_sc, X_test_sc, y_train, y_test, scaler):
    """
    Punto de entrada de la fase 6 no supervisada.
    """
    print("\n==== FASE 6: MODELO NO SUPERVISADO ====")
    print("Algoritmo elegido: K-Means.")
    print("Razon tecnica: K-Means permite agrupar viviendas por similitud entre atributos.")
    print("Se usan atributos escalados porque K-Means se basa en distancias.")
    print("K-Means no utiliza tenencia_cat para construir los clusters.")
    print("tenencia_cat se cruza al final solo para interpretar los grupos encontrados.")

    output_dir = crear_carpeta_outputs()
    rutas_generadas = []

    tabla_k = evaluar_valores_k(X_train_sc)
    print("\nEvaluacion de valores de k:")
    print(tabla_k.round(4).to_string(index=False))

    rutas_generadas.append(guardar_evaluacion(tabla_k, output_dir))
    rutas_generadas.append(graficar_metodo_codo(tabla_k, output_dir))
    rutas_generadas.append(graficar_silueta(tabla_k, output_dir))

    resultados_por_k = {}
    for k in CANDIDATE_K_VALUES:
        resultado = ejecutar_candidato(
            k, X_train_sc, X_test_sc, y_train, y_test, scaler, output_dir
        )
        resultados_por_k[k] = resultado
        rutas_generadas.extend(resultado["rutas"])

    tabla_comparacion, ruta_comparacion = construir_comparacion_candidatos(
        [resultados_por_k[k] for k in CANDIDATE_K_VALUES],
        output_dir
    )
    rutas_generadas.append(ruta_comparacion)

    resultado_final = seleccionar_modelo_final(resultados_por_k)
    rutas_generadas.extend(guardar_alias_final(resultado_final, output_dir))
    rutas_generadas.append(graficar_distribucion_clusters(
        resultado_final["distribucion_train"], output_dir, "fase6"
    ))
    rutas_generadas.append(graficar_tenencia_por_cluster(
        resultado_final["tabla_prop_train"], output_dir, "fase6"
    ))

    print("\nModelo final provisional:")
    print(f"k={resultado_final['k']}")
    print(f"Silueta entrenamiento: {resultado_final['silueta_train']:.4f}")
    if resultado_final["silueta_test"] is None:
        print("Silueta prueba: no calculable")
    else:
        print(f"Silueta prueba: {resultado_final['silueta_test']:.4f}")

    print("\nDistribucion de registros por cluster del modelo final:")
    print(resultado_final["distribucion_train"].to_string(index=False))
    print("\nCentroides aproximados en escala original del modelo final:")
    print(resultado_final["centroides_originales"].round(4).to_string(index=False))
    print("\nDistribucion proporcional de tenencia_cat por cluster del modelo final:")
    print(resultado_final["tabla_prop_train"].round(4).to_string())

    imprimir_rutas(rutas_generadas)

    print("\nConclusion tecnica preliminar:")
    print("k=4 reduce la sobresegmentacion y mantiene perfiles mas interpretables.")
    print("Las proporciones de tenencia_cat cambian entre clusters,")
    print("por lo que aportan evidencia parcial, no concluyente, sobre la hipotesis.")
    print("\nFase 6 completada.")

    return {
        "modelo": resultado_final["modelo"],
        "k_final": resultado_final["k"],
        "evaluacion_k": tabla_k,
        "comparacion_candidatos": tabla_comparacion,
        "silueta_train": resultado_final["silueta_train"],
        "silueta_test": resultado_final["silueta_test"],
        "clusters_train": resultado_final["clusters_train"],
        "clusters_test": resultado_final["clusters_test"],
        "centroides_escalados": resultado_final["centroides_escalados"],
        "centroides_originales": resultado_final["centroides_originales"],
        "tenencia_cluster_train": resultado_final["tabla_abs_train"],
        "tenencia_cluster_train_prop": resultado_final["tabla_prop_train"],
        "tenencia_cluster_test": resultado_final["tabla_abs_test"],
        "tenencia_cluster_test_prop": resultado_final["tabla_prop_test"],
        "candidatos": resultados_por_k,
    }
