import cv2
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
from deepface import DeepFace

# Configuración de la Interfaz (Modo oscuro para reducir fatiga visual)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ATFERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ATFER - Reconocimiento de Emociones")
        self.root.geometry("1000x700")
        
        # Variables de estado
        self.current_emotion = "Neutral"
        self.is_running = True
        self.video_capture = cv2.VideoCapture(0) # Iniciar cámara (Hardware) [cite: 147]

        # Configuración de Emojis y Colores según PDF 
        # Verde = Positivo, Rojo = Alerta/Enojo, Azul = Neutral/Tristeza
        self.emotion_map = {
            "happy":    {"emoji": "😊", "text": "FELICIDAD", "color": "#2CC985"}, # Verde
            "angry":    {"emoji": "😠", "text": "ENOJO",     "color": "#FF4B4B"}, # Rojo
            "sad":      {"emoji": "😢", "text": "TRISTEZA",  "color": "#3B8ED0"}, # Azul
            "fear":     {"emoji": "😨", "text": "MIEDO",     "color": "#E0A800"}, # Amarillo (Alerta)
            "surprise": {"emoji": "😲", "text": "SORPRESA",  "color": "#E0A800"},
            "neutral":  {"emoji": "😐", "text": "NEUTRAL",   "color": "#3B8ED0"},
        }

        self.setup_ui()
        self.process_video()

    def setup_ui(self):
        """Configura la interfaz gráfica dividida en dos paneles."""
        
        # --- Panel Izquierdo (Video) ---
        self.video_frame = ctk.CTkFrame(self.root, width=640, height=480, corner_radius=15)
        self.video_frame.pack(side="left", padx=20, pady=20, expand=True)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Panel Derecho (Feedback Accesible) ---
        self.info_frame = ctk.CTkFrame(self.root, width=300, corner_radius=15)
        self.info_frame.pack(side="right", fill="y", padx=20, pady=20)

        # Título del Proyecto
        self.title_label = ctk.CTkLabel(
            self.info_frame, 
            text="ATFER", 
            font=("Roboto Medium", 30)
        )
        self.title_label.pack(pady=(40, 20))

        # Emoji Gigante (Feedback visual inmediato) [cite: 40]
        self.emoji_label = ctk.CTkLabel(
            self.info_frame, 
            text="😐", 
            font=("Segoe UI Emoji", 100) # Fuente compatible con emojis
        )
        self.emoji_label.pack(pady=20)

        # Texto de la Emoción
        self.emotion_text_label = ctk.CTkLabel(
            self.info_frame, 
            text="NEUTRAL", 
            font=("Roboto", 24, "bold"),
            text_color="#3B8ED0"
        )
        self.emotion_text_label.pack(pady=10)

        # Botón de Salir
        self.quit_button = ctk.CTkButton(
            self.info_frame, 
            text="Salir", 
            command=self.close_app,
            fg_color="#FF4B4B",
            hover_color="#D13434"
        )
        self.quit_button.pack(side="bottom", pady=40)

    def analyze_emotion(self, frame):
        """
        Función que se ejecuta en un hilo separado para no congelar la UI.
        Utiliza DeepFace para analizar la emoción del frame actual.
        """
        try:
            # DeepFace requiere la imagen en formato BGR (OpenCV standard)
            # actions=['emotion'] optimiza para solo buscar emociones
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='opencv' # Backend rápido
            )
            
            # Obtener la emoción dominante
            dominant_emotion = result[0]['dominant_emotion']
            self.update_emotion_display(dominant_emotion)
            
        except Exception as e:
            # Si no detecta rostro o hay error, asume neutral
            pass

    def update_emotion_display(self, emotion):
        """Actualiza el emoji, color y texto en la interfaz."""
        data = self.emotion_map.get(emotion, self.emotion_map["neutral"])
        
        # Actualizar UI de forma segura
        self.emoji_label.configure(text=data["emoji"])
        self.emotion_text_label.configure(text=data["text"], text_color=data["color"])
        
        # Borde de color alrededor del video para refuerzo visual 
        self.video_frame.configure(border_width=4, border_color=data["color"])

    def process_video(self):
        """Bucle principal de captura de video y actualización de frames."""
        if not self.is_running:
            return

        ret, frame = self.video_capture.read()
        if ret:
            # Espejo (efecto espejo es más natural para el usuario)
            frame = cv2.flip(frame, 1)

            # 1. Análisis de IA:
            # Ejecutamos el análisis solo cada 10 frames para mejorar rendimiento
            # (Simula tiempo real sin sobrecargar el procesador)
            if getattr(self, "frame_count", 0) % 15 == 0:
                threading.Thread(target=self.analyze_emotion, args=(frame,), daemon=True).start()
            
            self.frame_count = getattr(self, "frame_count", 0) + 1

            # 2. Conversión de imagen para Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Llamar a esta función de nuevo después de 10ms
        self.root.after(10, self.process_video)

    def close_app(self):
        """Cierra la cámara y la ventana correctamente."""
        self.is_running = False
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ATFERApp(root)
    root.mainloop()