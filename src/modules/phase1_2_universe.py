import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def cargar_y_filtrar(ruta_csv, cve_estado, nombre_estado):
    """
    Carga vivienda.csv, filtra por entidad federativa y
    retorna el dataframe filtrado.
    """
    df = pd.read_csv(ruta_csv, encoding='latin-1', low_memory=False)
    print(f"Registros totales: {len(df):,}")
    print(f"Columnas        : {df.shape[1]}")

    df['cve_estado'] = (
        df['ubica_geo']
        .astype(str)
        .str.zfill(5)
        .str[:2]
    )

    df_estado = df[df['cve_estado'] == cve_estado].copy()
    print(f"\nRegistros en {nombre_estado}: {len(df_estado):,}")

    return df_estado


def explorar_datos(df_estado, nombre_estado):
    """
    Imprime vista general del dataframe filtrado.
    """
    print("\nPrimeras filas:")
    print(df_estado.head(3))
    print("\nTipos de datos:")
    print(df_estado.dtypes)
    print("\nEstadísticas descriptivas:")
    print(df_estado.describe())


def matriz_correlacion_general(df_estado, nombre_estado):
    """
    Genera y guarda la matriz de correlación de todas
    las columnas numéricas (excluyendo identificadores).
    Retorna el dataframe de pares ordenado por correlación.
    """
    cols_excluir = [
        'folioviv', 'ubica_geo', 'cve_estado',
        'upm', 'factor', 'est_dis', 'est_socio'
    ]
    cols_num = [
        c for c in df_estado.select_dtypes(include='number').columns
        if c not in cols_excluir
    ]

    corr_matrix = df_estado[cols_num].corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1,
        linewidths=0.3, annot_kws={'size': 7}, ax=ax
    )
    ax.set_title(
        f'Matriz de Correlación — Viviendas {nombre_estado}\n'
        f'(n = {len(df_estado):,} registros)',
        fontsize=13, pad=15
    )
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('correlacion_general.png', dpi=150)
    plt.show()

    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ['var1', 'var2', 'correlacion']
    corr_pairs['abs_corr'] = corr_pairs['correlacion'].abs()

    print("\n── Top 15 correlaciones más fuertes ──")
    print(
        corr_pairs
        .sort_values('abs_corr', ascending=False)
        .head(15)
        .to_string(index=False)
    )

    return corr_pairs


def matriz_correlacion_seleccion(df_estado, atributos, nombre_estado):
    """
    Genera la matriz de correlación solo con los atributos
    seleccionados e imprime su relación contra la etiqueta.
    """
    df_sel = df_estado[atributos].copy()
    corr_sel = df_sel.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_sel,
        annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, annot_kws={'size': 11},
        square=True, ax=ax
    )
    ax.set_title(
        f'Correlación — Atributos Seleccionados\n{nombre_estado}',
        fontsize=12, pad=12
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('correlacion_seleccion.png', dpi=150)
    plt.show()

    print("\n── Correlación de atributos vs etiqueta (tenencia) ──")
    corr_vs_label = (
        corr_sel['tenencia']
        .drop('tenencia')
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    print(corr_vs_label.to_string())


def ejecutar(ruta_csv, cve_estado, nombre_estado, atributos):
    """
    Punto de entrada del módulo 1.
    Retorna el dataframe filtrado listo para el módulo 3.
    """
    df_estado = cargar_y_filtrar(ruta_csv, cve_estado, nombre_estado)
    explorar_datos(df_estado, nombre_estado)
    matriz_correlacion_general(df_estado, nombre_estado)
    matriz_correlacion_seleccion(df_estado, atributos, nombre_estado)

    return df_estado