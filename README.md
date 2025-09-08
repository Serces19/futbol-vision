# Sistema de Detección y Seguimiento de Jugadores para Análisis Táctico en Fútbol

## 📖 Introducción

Este proyecto presenta una solución avanzada de visión por computadora para automatizar la extracción de datos posicionales de jugadores y del balón a partir de videos de partidos de fútbol grabados con una cámara general.

En el fútbol moderno, el análisis táctico y el rendimiento de los jugadores son campos de estudio que dependen en gran medida de datos precisos. La obtención manual de estos datos es un proceso costoso y, a menudo, inviable. Nuestro sistema aborda este problema al transformar el metraje de video en datos estructurados y cuantificables, lo que permite a entrenadores, analistas y científicos de datos obtener información valiosa para la toma de decisiones.

## Características Principales

* **Detección de Objetos:** Utiliza un modelo **YOLOv8** ajustado para identificar con precisión a jugadores, porteros, árbitros y el balón.
* **Tracking de Jugadores:** Implementa el algoritmo **ByteTrack** para seguir de forma individual y consistente a cada jugador a lo largo del video.
* **Clasificación de Equipos:** Emplea un modelo **Osnet** para generar embeddings de las imágenes de los jugadores, que luego son clasificados por equipo usando el algoritmo **K-means**.
* **Mapeo del Campo de Juego:** Utiliza la librería **PnLCalib** para calibrar el campo y estimar los parámetros de la cámara.
* **Homografía:** Genera una matriz de homografía para proyectar las posiciones 2D de los jugadores de la imagen a su posición real en el campo de juego.
* **Visualización:** Genera un mapa 2D del campo y superpone anotaciones sobre el video original para una visualización clara del análisis.

## Guía de Inicio

### Requisitos

* Python 3.10+
* GPU (recomendado para un rendimiento óptimo)

### Opción 1: Google Colab (Recomendada)

La forma más rápida de probar el sistema es a través de nuestro Google Colab, que ya viene preconfigurado con las dependencias y la GPU necesarias.

* [**Abrir en Google Colab**](https://colab.research.google.com/drive/1fFRrtbZIwJvD_jcA8yWk7GtXhGO6fRKR?usp=sharing)

### Opción 2: Instalación Local

1.  **Descarga el proyecto completo** (9 GB) desde Google Drive:
    * [**Descargar desde Google Drive**](https://drive.google.com/drive/folders/1nP-isGYyQWVDnW2rlkxFp80v3_w38OvL?usp=sharing)

2.  **Crea un entorno virtual e instala las dependencias principales:**
    ```bash
    git clone https://github.com/Serces19/futbol-vision.git
    cd futbol-vision
    uv venv futbol-vision --python 3.10
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

3.  **Instala las dependencias específicas de ByteTrack:**
    ```bash
    cd ByteTrack
    uv pip install -r requirements.txt
    uv pip install -v -e .
    cd ..
    ```

4.  **Instala las dependencias de PnLCalib:**
    ```bash
    cd PnLCalib
    uv pip install -r requirements.txt
    cd ..
    ```

## Uso

Una vez que tengas todas las dependencias instaladas, puedes ejecutar el script principal para procesar un video:

```bash
python analisis_completo_video.py
```




## Arquitectura Técnica
El proyecto se compone de los siguientes módulos interconectados para lograr el resultado final:

#### Detección de Objetos
Se utilizó un modelo YOLOv8m que fue ajustado con un dataset propio y uno de Roboflow. El modelo final, best_v02.pt, detecta las clases Ball, Players, Goalkeeper y Referee con alta precisión, lo que es crucial para el rendimiento de los módulos posteriores.

#### Tracking de Objetos
Se implementó ByteTrack, un algoritmo de seguimiento que utiliza la confianza de las detecciones de YOLO para seguir a los objetos a través de los frames. El repositorio original de ByteTrack fue modificado para su correcta integración.

#### Clasificación con K-means
Para diferenciar entre equipos, las imágenes de los jugadores son procesadas por el modelo Osnet (entrenado en prendas de vestir) para generar embeddings. Estos embeddings son luego clasificados en dos grupos por el algoritmo K-means. El tracking de ByteTrack se usa para mantener la consistencia en la clasificación.

#### Calibración y Detección del Campo
El módulo PnLCalib es responsable de la detección del campo de juego. Utiliza un modelo de segmentación para identificar las líneas y otro para inferir los puntos clave del campo, lo que permite estimar los parámetros de la cámara.

#### Homografía
Con los parámetros de la cámara obtenidos, se crea una matriz de homografía que sirve para transformar las coordenadas 2D de los jugadores en la imagen a sus coordenadas reales en el plano 2D del campo de juego, permitiendo el análisis posicional.

### Créditos
YOLOv8: Repositorio Oficial

ByteTrack: Repositorio utilizado

PnLCalib: Repositorio Oficial