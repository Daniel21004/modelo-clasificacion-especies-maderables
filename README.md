# modelo-clasificacion-especies-maderables
Nota: Todo el código permanece dentro de Base code con la siguiente estructura
- Gráficas -> Contiene imágenes correspondientes al desempeño de los modelos
- Extrapolación -> Recursos relacionados a la extrapolación del modelo
- Entorno abierto -> Recursos relacionados a la evaluación del modelo en un entorno abierto
- Estudio de Ablacion -> El código utilizado para desarrollar el estudio de ablación
- Archivos .pt -> En concreto dos: Modelo con aumento de datos y sin atención; y Modelo con aumento de datos y con atención
- TriAttentionArchitectura -> Código que contiene la arquitectura de TA, utilizada en el estudio de ablación
- codigo final -> Codigo completo para el entrenamiento de los modelos, incluido "Modelo con aumento de datos y con atención"
- metadata_example -> Archivo .csv que contiene el cuerpo general para la lectura de las imágenes. El dataset original se encuentra subido en Zenodo: [Dataset de ablación](https://doi.org/10.5281/zenodo.17958137) y [Dataset principal](https://doi.org/10.5281/zenodo.18048274). Se recomienda observar la estructura de los dos Dataset

# Instalar dependencias
Para instalar dependencias, primero cree un entorno virtual en python y continue con la instalación del archivo "requirements.txt"

## Entorno virtual
```
python3 -m venv .venv
```
(No se olvide de activar el mismo, consulte externamente como activarlo dependiendo de su SO: MAC, Linux o Windows)

## Instalación dependencias
```
pip install -r requirements.txt
```

Una vez instaladas las dependencias, ya podrá ejecutar el proyecto.
Cabe recalcar que TriAttentionArchitecture.py solo contiene la arquitectura que necesita Estudio_ablacion.py. Los archivos de interés son Estudio_ablacion.py y codigo_final.ipynb.
Igualmente recuerde que los dos necesitan un metadata.csv con estructura diferente, por ello, se recomienda observa la estructura de los dos Dataset antes mencionados

## Métricas del modelo
- ACC1: 0.81
- ACC5: 1.0
- Número de parámetros: 4.2M
- Peso: 16 MB
- GFLOPS: 0.9GFLOPS
- Tiempo inferencia CPU: 77.589MS
- 
