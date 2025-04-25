import cv2
import mediapipe as mp
import numpy as np
import serial

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Inicializar captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Inicializar conexión serial con Arduino (ajusta el puerto según tu configuración)
ser = serial.Serial('COM3', 9600, timeout=1)

# Definir pares de landmarks para punta y base de cada dedo
dedos = [
    (4, 2),   # Pulgar
    (8, 5),   # Índice
    (12, 9),  # Medio
    (16, 13), # Anular
    (20, 17)  # Meñique
]

# Variable para rastrear el último número enviado
last_num_dedos = -1

# Bucle principal
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

            # Obtener posición de la muñeca (landmark 0)
            wrist = hand_landmarks.landmark[0]
            wrist_pos = np.array([wrist.x * w, wrist.y * h])

            # Obtener posición de la base del dedo medio (landmark 9)
            middle_base = hand_landmarks.landmark[9]
            middle_base_pos = np.array([middle_base.x * w, middle_base.y * h])

            # Calcular distancia de referencia
            dist_ref = np.linalg.norm(wrist_pos - middle_base_pos)

            # Umbrales proporcionales al tamaño de la mano
            umbral_y = 0.1 * dist_ref  # Para dedos índice a meñique
            umbral_pulgar = 0.5 * dist_ref  # Para el pulgar

            # Dibujar landmarks en la mano
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                mp_draw.DrawingSpec(color=(255, 0, 255), thickness=4)
            )

            # Contar dedos levantados
            # Primero, manejar el pulgar (dedos[0])
            thumb_tip = hand_landmarks.landmark[4]  # Punta del pulgar
            index_base = hand_landmarks.landmark[5]  # Base del índice
            thumb_tip_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
            index_base_pos = np.array([index_base.x * w, index_base.y * h])
            dist_thumb_index = np.linalg.norm(thumb_tip_pos - index_base_pos)

            # Si la distancia es grande, el pulgar está levantado
            if dist_thumb_index > umbral_pulgar:
                num_dedos += 1

            # Luego, los otros dedos (índice a meñique)
            for tip, base in dedos[1:]:  # Excluyendo el pulgar
                tip_landmark = hand_landmarks.landmark[tip]
                base_landmark = hand_landmarks.landmark[base]
                tip_y = tip_landmark.y * h
                base_y = base_landmark.y * h

                # Verificar si la punta está suficientemente por encima de la base
                if tip_y < base_y - umbral_y:
                    num_dedos += 1

    # Mostrar conteo en pantalla
    cv2.putText(frame, f"Dedos levantados: {num_dedos}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Enviar a Arduino solo si cambió el número
    if num_dedos != last_num_dedos:
        ser.write(str(num_dedos).encode())
        last_num_dedos = num_dedos

    # Mostrar video
    cv2.imshow("Conteo de Dedos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
ser.close()