import pandas as pd
import numpy as np

def validar_nulos(df, atributos):
    print("\n-- Validando Valores Nulos --")
    df_work = df[atributos].copy()
    nulos = df_work.isnull().sum()
    print(nulos)
    
    if nulos.sum() > 0:
        print("\nImputando valores nulos con la moda...")
        for col in atributos:
            if df_work[col].isnull().sum() > 0:
                moda = df_work[col].mode()[0]
                df_work[col].fillna(moda, inplace=True)
        print("Valores nulos corregidos.")
    else:
        print("El dataset no contiene valores nulos en estos atributos.")
        
    return df_work

def limpiar_texto_y_acentos(df, atributos):
    print("\n-- Limpiando Texto y Acentos --")
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
        print(f"Limpieza aplicada a las columnas: {list(columnas_texto)}")
    else:
        print("No se detectaron columnas de texto. No es necesaria la limpieza de acentos.")
        
    return df

def verificar_tipos_datos(df, atributos):
    print("\n-- Verificando Tipos de Datos para ML --")
    for col in atributos:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Convirtiendo la columna '{col}' a tipo numérico...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if df[col].isnull().sum() > 0:
                df[col].fillna(-1, inplace=True)
                
    print(df[atributos].dtypes)
    return df

"""
def validar_fase3(df_limpio):
    print("\n--- VALIDACIÓN FINAL FASE 3 ---")
    print("Dimensiones del dataset de trabajo:", df_limpio.shape)
    print("\nPrimeros registros procesados:")
    print(df_limpio.head())
"""
    

def ejecutar(df_estado, atributos):
    print("---- INICIANDO FASE 3: PRE-PROCESAMIENTO ----")
    
    df_limpio = validar_nulos(df_estado, atributos)
    df_limpio = limpiar_texto_y_acentos(df_limpio, atributos)    
    df_limpio = verificar_tipos_datos(df_limpio, atributos)

    print("\nPre-procesamiento completado. Dataset listo para particionamiento.")

    # validar_fase3(df_limpio)
    return df_limpio