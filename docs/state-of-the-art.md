#Redes utilizadas en reconocimiento y  clasificación de rostros:


1. HyperFace (Ranjan, 2017a)
2. All-In-One Face (Ranjan, 2017b)
3. A Jointly Learned Deep Architecture for Facial Attribute Analysis and Face Detection in the Wild (He et al., 2017)
4. DAGER: Deep Age, Gender and Emotion Recognition Using Convolutional Neural Network (Dehghan, 2017)
5. A Convolutional Neural Network for Gender Recognition Optimizing the Accuracy/Speed Tradeoff (Greco et al., 2020)
6. Deep Learning based approach to detect Customer Age, Gender and Expression in Surveillance Video (Ijjina, E. P., Kanahasabai, G., & Joshi, A. S., 2020)

## Redes utilizadas en reconocimiento y  clasificación de rostros::

2. All-In-One Face (Ranjan, 2017b)
Se inicia con la red preentrenada de Sankaranayanan et al. (2016) para reconocimiento facial como backbone. Consiste en 7 capas convolucionales y 3 capas fully connected.
Las 6 primeras capas convolucionales se utilizan para entrenar otras tareas, las que se dividen en 2 grupos: independientes (detección, visibilidad, keypoints, etc) y dependientes (género, edad) del sujeto.

Yolo Face Detection
Arquitectura Yolo aplicada para detectar rostro y género (además de otras características).

Pesos pre-entrenados disponibles en:
https://github.com/grausof/YoloKerasFaceDetection
https://github.com/OValery16/gender-age-classification

