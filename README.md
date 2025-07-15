# Pipeline for prediction pIC50 from SMILES strings.

This project was written originally to compete in the 2023 edition of the ENS Data Challenge "Predicting molecule-protein interaction for drug discovery". Although it is light enough to run on a laptop, it has been built with a more sophisticated production environment in mind. 

Conda is the recommended virtual environment manager for the code due to the RDKIT compatibility. However, RDKIT can be compiled from source and installed on a venv environment.


## How to run.
The process of fingerprint generation and parameter search is automated. One just needs to ensure that the dataset is in the correct 







## Características

* **Estructura Profesional:** Organizado como un paquete Python instalable y modular.
* **Reproducibilidad:** El entorno exacto está definido en `environment.yml` para ser recreado con Conda.
* **Configurabilidad:** Todos los parámetros del experimento (rutas, tipo de fingerprint, configuración de la optimización) se gestionan desde un único fichero `config.py`.
* **Trazabilidad:** Cada ejecución genera un `run_name` único para guardar un log en CSV con el historial de la búsqueda, un gráfico de convergencia y el modelo final entrenado.

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  **Crear y activar el entorno Conda:**
    El fichero `environment.yml` contiene todas las dependencias necesarias.
    ```bash
    conda env create -f environment.yml
    conda activate pic50_env
    ```

3.  **Instalar el proyecto en modo editable:**
    ```bash
    pip install -e .
    ```

## Uso

Para ejecutar el pipeline completo (optimización, entrenamiento y guardado), simplemente ejecuta el script principal desde la raíz del proyecto:
```bash
python train.py
```
Los resultados (logs, gráficos) se guardarán en la carpeta `/results` y el modelo final en `/models`, cada uno con un nombre de ejecución único.

## Estructura del Proyecto
```
├── config.py            # Fichero central de configuración
├── train.py             # Script principal para orquestar el entrenamiento
├── environment.yml      # Fichero de dependencias para Conda
├── setup.py             # Fichero de instalación del paquete
├── pic50_predictor/     # Paquete principal con la lógica de la aplicación
│   ├── __init__.py
│   ├── features.py      # Funciones para carga de datos y featurización
│   └── model.py         # Clases y funciones para el modelo y su optimización
├── models/              # Directorio para los modelos entrenados
└── results/             # Directorio para los logs y gráficos de resultados
```
