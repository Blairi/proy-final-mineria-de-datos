import pandas as pd
import numpy as np


def validar_nulos(df, atributos):
    print("\n── Validando valores nulos ──")
    df_work = df[atributos].copy()
    nulos = df_work.isnull().sum()
    print(nulos)

    if nulos.sum() > 0:
        print("\nImputando valores nulos con la moda...")
        for col in atributos:
            if df_work[col].isnull().sum() > 0:
                moda = df_work[col].mode()[0]
                df_work[col] = df_work[col].fillna(moda)
                # Justificación: se usa la moda porque los atributos
                # seleccionados son categóricos o binarios, donde la
                # media no tiene significado semántico.
        print("Valores nulos corregidos.")
    else:
        print("Sin valores nulos en los atributos seleccionados.")

    return df_work


def limpiar_texto_y_acentos(df, atributos):
    print("\n── Limpiando texto y acentos ──")
    columnas_texto = df[atributos].select_dtypes(include=['object']).columns

    if len(columnas_texto) > 0:
        for col in columnas_texto:
            df[col] = (
                df[col].astype(str)
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
                .str.lower()
                .str.strip()
            )
        print(f"Limpieza aplicada a: {list(columnas_texto)}")
    else:
        print("No hay columnas de texto. Limpieza de acentos no necesaria.")

    return df


def verificar_tipos_datos(df, atributos):
    print("\n── Verificando tipos de datos ──")
    for col in atributos:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Convirtiendo '{col}' a numérico...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().sum() > 0:
                # Usamos moda en lugar de -1 para no introducir
                # un valor fuera de rango que confunda al modelo
                moda = df[col].mode()[0]
                df[col] = df[col].fillna(moda)

    print(df[atributos].dtypes)
    return df


def discretizar_tenencia(df):
    """
    c. Discretización de 'tenencia'
    Razón: el campo es un código numérico sin orden matemático
    (1=propia pagada, 2=propia pagándose, 3=rentada, 4=prestada,
    5=otra). El algoritmo supervisado necesita una etiqueta
    categórica legible. Se agrupa en 3 clases para balancear
    la distribución y simplificar la clasificación.
    """
    print("\n── Discretizando etiqueta 'tenencia' ──")   # ← 4 espacios
    df['tenencia'] = pd.to_numeric(df['tenencia'], errors='coerce')
    df['tenencia'] = df['tenencia'].fillna(df['tenencia'].mode()[0])
    df['tenencia'] = df['tenencia'].astype(int)

    print("\nValores únicos de tenencia ANTES del map:")
    print(df['tenencia'].value_counts(dropna=False))

    mapa = {1: 'Propia', 2: 'Propia', 3: 'Rentada', 4: 'Prestada', 5: 'Prestada'}
    df['tenencia_cat'] = df['tenencia'].map(mapa)
    df = df.drop(columns=['tenencia'])

    nulos_post_map = df['tenencia_cat'].isnull().sum()
    if nulos_post_map > 0:
        print(f"{nulos_post_map} valores no mapeados:")
        print(df.loc[df['tenencia_cat'].isnull(), 'tenencia_cat'].value_counts())
        df['tenencia_cat'] = df['tenencia_cat'].fillna('Otra')

    print("Distribución de clases:")
    print(df['tenencia_cat'].value_counts())

    return df


def validar_resultado(df):
    print("\n── Validación final fase 3 ──")
    print(f"Shape del dataset: {df.shape}")
    print(f"Nulos restantes  : {df.isnull().sum().sum()}")
    print("\nPrimeros registros procesados:")
    print(df.head())


def ejecutar(df_estado, atributos):
    print("==== FASE 3: PRE-PROCESAMIENTO ====")

    print(f"Columnas recibidas: {list(df_estado.columns)}")  # diagnóstico
    print(f"ATRIBUTOS         : {atributos}")                # diagnóstico

    df_limpio = validar_nulos(df_estado, atributos)
    df_limpio = limpiar_texto_y_acentos(df_limpio, atributos)
    df_limpio = verificar_tipos_datos(df_limpio, atributos)

    print(f"Columnas antes de discretizar: {list(df_limpio.columns)}")

    df_limpio = discretizar_tenencia(df_limpio)
    validar_resultado(df_limpio)

    print("\nPre-procesamiento completado. Dataset listo para módulo 4.")
    return df_limpio