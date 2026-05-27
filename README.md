# Proyecto Final: Minería de Datos 2026-2 📊

## Descripción del Proyecto
[cite_start]Este repositorio contiene el desarrollo práctico y teórico para demostrar los conocimientos adquiridos en la materia de Minería de Datos[cite: 6]. 

[cite_start]El proyecto sigue la metodología KDD (Knowledge Discovery in Databases) para analizar los datos de la **Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2024** del INEGI, trabajando específicamente con el archivo `vivienda.csv`[cite: 12, 14]. [cite_start]El análisis se centra en una sola entidad federativa para realizar predicción de casos y descubrimiento de patrones[cite: 12, 15].

## Equipo de Desarrollo
* Axel Fernando Montiel Aviles
* 
*
* 

## Objetivos Técnicos
Se implementarán dos enfoques principales de Machine Learning:

1. [cite_start]**Modelo Supervisado:** Predicción basada en un atributo categórico (etiqueta)[cite: 16]. [cite_start]Se evaluará el rendimiento del modelo utilizando la matriz de confusión y extrayendo métricas clave como Accuracy, Precision, Recall y F1-Score[cite: 43, 44, 47, 48, 50, 52]. [cite_start]Además, se generará la curva ROC[cite: 54].
2. [cite_start]**Modelo No Supervisado (Clusterización):** Agrupamiento de datos utilizando el método del codo para determinar el número de clusters[cite: 66, 69]. [cite_start]Los resultados se validarán internamente utilizando la métrica de Silueta[cite: 72].

---

## 🚀 Guía de Arranque Rápido para el Equipo

Para estandarizar el entorno de desarrollo y evitar conflictos con las dependencias, sigan estos pasos al clonar el proyecto por primera vez.

### 1. Clonar el repositorio
Abre tu terminal y clona este proyecto en tu máquina local:
```bash
git clone https://github.com/Blairi/proy-final-mineria-de-datos
cd proy-final-mineria-de-datos
```

Los datos los descargaremos de aquí:
[descarga desde SiCAAD](https://www.siccaad.unam.mx/students/message_list/download/2574/)

Descomprimimos y los dejamos en la raíz del proyecto.

### 2. Configurar el Entorno Virtual

Es obligatorio usar un entorno aislado para no afectar la instalación global de Python de tu sistema operativo.

En Linux (ej. Fedora) / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows (Command Prompt / PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias

Asegúrate de que el entorno esté activado (verás un (venv) en tu terminal). El proyecto utiliza versiones específicas para garantizar la estabilidad de las métricas de scikit-learn y gráficas.

El archivo requirements.txt del repositorio contiene lo siguiente:

```Plaintext
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.1.post1
jupyter==1.0.0
notebook==7.1.2
```

Necesitamos asegurarnos de que tenemos instalado las ultimas versiones de python y pip. Para ello necesitamos ejecutar (en Linux):
```Bash
sudo dnf install python3-devel
```

```Bash
pip install --upgrade pip
```

Instalamos todo ejecutando:
```Bash
pip install -r requirements.txt
```

## Ejecutar programa
Ejecutar desde la raíz del proyecto:
```Bash
python3 src/main.py
```

Importante ejecutar desde la raiz para no tener problemas con la lectura del archivo csv.