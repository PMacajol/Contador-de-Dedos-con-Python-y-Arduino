Conteo de Dedos Levantados con MediaPipe y Comunicación Serial con Arduino

Este proyecto implementa un sistema que utiliza la biblioteca MediaPipe para detectar y rastrear manos en tiempo real mediante una cámara web, contar el número de dedos levantados y enviar esa información a un Arduino a través de una conexión serial. El propósito es crear una interfaz hombre-máquina simple donde el conteo de dedos pueda usarse para controlar dispositivos conectados al Arduino.

## Requisitos Previos

- **Python 3.x** instalado.
- Bibliotecas de Python:
  - opencv-python`: Para captura y procesamiento de video.
  - mediapipe`: Para detección y rastreo de manos.
  - numpy`: Para cálculos numéricos.
  - pyserial`: Para comunicación serial con Arduino.
- Una cámara web conectada al ordenador.
- Un Arduino conectado mediante un puerto serial (ejemplo: COM3).

### Instalación de Dependencias

    pip install opencv-python mediapipe numpy pyserial

Estructura del Código
A continuación, se detalla cada sección del código, explicando qué hace y por qué se utiliza.

             import cv2
            import mediapipe as mp
            import numpy as np
            import serial

Por qué lo usamos:
      - cv2 (OpenCV): Captura video desde la cámara y procesa imágenes (volteo, texto en pantalla).
      - mediapipe : Proporciona herramientas para detectar y rastrear manos (mp.solutions.hands) y dibujar landmarks (mp.solutions.drawing_utils).
      - numpy: Realiza cálculos numéricos como distancias euclidianas para determinar si un dedo está levantado.
      - serial: Establece comunicación serial con el Arduino para enviar el conteo de dedos.

2. Inicialización de MediaPipe
   
          mp_hands = mp.solutions.hands
          mp_draw = mp.solutions.drawing_utils
          hands = mp_hands.Hands(
              max_num_hands=1,
              min_detection_confidence=0.8,
              min_tracking_confidence=0.8
          )
   
•	Qué hace:
    o	mp_hands: Accede al módulo de detección de manos de MediaPipe.
    o	mp_draw: Proporciona funciones para visualizar landmarks y conexiones de la mano.
    o	hands: Crea un objeto para procesar manos con parámetros específicos.
•	Por qué lo usamos:
    o	max_num_hands=1: Limita la detección a una mano para simplificar el procesamiento.
    o	min_detection_confidence=0.8: Requiere un 80% de confianza para detectar una mano, asegurando precisión.
    o	min_tracking_confidence=0.8: Requiere un 80% de confianza para rastrear la mano, manteniendo estabilidad.

3. Inicialización de la Captura de Video
   
           cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            exit()
•	Qué hace:
    o	cap: Inicia la captura de video desde la cámara predeterminada (índice 0).
    o	Verifica si la cámara se abrió correctamente; si no, termina el programa.
•	Por qué lo usamos:
    o	Necesitamos capturar video en tiempo real para procesar las imágenes de la mano.
    o	La verificación evita errores si la cámara no está disponible.

4. Inicialización de la Conexión Serial con Arduino

       ser = serial.Serial('COM3', 9600, timeout=1)

•	Qué hace:
    o	Establece una conexión serial en el puerto COM3 a 9600 baudios con un timeout de 1 segundo.
•	Por qué lo usamos:
    o	Permite enviar el número de dedos levantados al Arduino para controlar dispositivos externos.
    o	Nota: El puerto COM3 debe ajustarse según el sistema (ejemplo: /dev/ttyUSB0 en Linux).

5. Definición de Pares de Landmarks para Dedos

              dedos = [
            (4, 2),   # Pulgar
            (8, 5),   # Índice
            (12, 9),  # Medio
            (16, 13), # Anular
            (20, 17)  # Meñique
        ]

•	Qué hace:
    o	Define pares de índices de landmarks (punta y base) para cada dedo según el modelo de MediaPipe.
•	Por qué lo usamos:
    o	Estos pares permiten comparar las posiciones de la punta y la base de cada dedo para determinar si está levantado.
    o	MediaPipe asigna 21 landmarks por mano (0-20), y estos índices son estándar para cada dedo.


6. Variable para Rastrear el Último Número Enviado

       last_num_dedos = -1

   •	Qué hace:
        o	Almacena el último número de dedos enviados al Arduino.
•	Por qué lo usamos:
        o	Evita enviar datos repetidos al Arduino, reduciendo la carga de comunicación.

7. Bucle Principal para Procesamiento de Video

       while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
    
        num_dedos = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape

            # Obtener posición de la muñeca y base del dedo medio
            wrist = hand_landmarks.landmark[0]
            wrist_pos = np.array([wrist.x * w, wrist.y * h])
            middle_base = hand_landmarks.landmark[9]
            middle_base_pos = np.array([middle_base.x * w, middle_base.y * h])

            # Calcular distancia de referencia
            dist_ref = np.linalg.norm(wrist_pos - middle_base_pos)

            # Umbrales proporcionales
            umbral_y = 0.1 * dist_ref
            umbral_pulgar = 0.5 * dist_ref

            # Dibujar landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                mp_draw.DrawingSpec(color=(255, 0, 255), thickness=4)
            )

            # Contar dedos levantados
            thumb_tip = hand_landmarks.landmark[4]
            index_base = hand_landmarks.landmark[5]
            thumb_tip_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
            index_base_pos = np.array([index_base.x * w, index_base.y * h])
            dist_thumb_index = np.linalg.norm(thumb_tip_pos - index_base_pos)
            if dist_thumb_index > umbral_pulgar:
                num_dedos += 1

            for tip, base in dedos[1:]:
                tip_landmark = hand_landmarks.landmark[tip]
                base_landmark = hand_landmarks.landmark[base]
                tip_y = tip_landmark.y * h
                base_y = base_landmark.y * h
                if tip_y < base_y - umbral_y:
                    num_dedos += 1

        # Mostrar conteo en pantalla
        cv2.putText(frame, f"Dedos levantados: {num_dedos}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Enviar a Arduino si cambió
        if num_dedos != last_num_dedos:
            ser.write(str(num_dedos).encode())
            last_num_dedos = num_dedos
    
        # Mostrar video
        cv2.imshow("Conteo de Dedos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   •	Qué hace:
        1.	Captura y Preprocesamiento:
        	Lee frames de la cámara (cap.read()).
        	Voltea el frame horizontalmente (cv2.flip) para efecto espejo.
        	Convierte de BGR a RGB (cv2.cvtColor) porque MediaPipe requiere RGB.
        2.	Detección de Manos:
        	Procesa el frame con hands.process() para obtener landmarks.
        3.	Conteo de Dedos:
        	Calcula una distancia de referencia (dist_ref) entre la muñeca y la base del dedo medio para umbrales adaptativos.
        	Define umbrales proporcionales para el pulgar y otros dedos.
        	Dibuja landmarks en el frame para visualización.
        	Evalúa el pulgar comparando la distancia entre su punta y la base del índice.
        	Evalúa los otros dedos comparando posiciones Y de punta y base.
        4.	Visualización y Comunicación:
        	Muestra el conteo en pantalla (cv2.putText).
        	Envía el conteo al Arduino si cambia (ser.write).
        	Muestra el frame procesado (cv2.imshow) y permite salir con 'q'.
•	Por qué lo usamos:
        o	Procesar video en tiempo real permite una interacción dinámica.
        o	Los umbrales adaptativos aseguran precisión independientemente del tamaño de la mano.
        o	La visualización ayuda a depurar y entender el comportamiento.
        o	La comunicación eficiente con Arduino optimiza el control de dispositivos.


8. Liberación de Recursos
   
                         cap.release()
                      cv2.destroyAllWindows()
                      ser.close()

   •	Qué hace:
        o	Libera la cámara, cierra ventanas y termina la conexión serial.
•	Por qué lo usamos:
        o	Evita que los recursos queden bloqueados tras finalizar el programa.


9. VIDEO DEMOSTRATIVO:
      https://drive.google.com/file/d/1eAOXSjKv0yyiSc_cpLXelSk4KMa4YhNX/view?usp=sharing
   
