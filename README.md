# ATFER - Autism-Friendly Tool for Facial Expression Recognition

> **"Descifra las emociones, fortalece las conexiones."**

**ATFER** es una herramienta tecnológica diseñada para promover la inclusión y la autonomía de personas dentro del espectro autista. Utilizando Inteligencia Artificial y principios de *Physical Computing*, el sistema detecta expresiones faciales en tiempo real y las traduce en señales visuales claras (emojis y códigos de color) para facilitar la interpretación emocional y reducir la ansiedad social.

Este proyecto fue desarrollado como parte del **Club de Computación Física (PHYCOM) - 2025 T-1**.

## 🚀 Características Principales

**Reconocimiento en Tiempo Real:** Detecta emociones básicas (alegría, enojo, tristeza, sorpresa, etc.) usando la cámara del dispositivo.
**Feedback Accesible:** Interfaz amigable que traduce la emoción a un emoji grande y un color representativo (Verde=Positivo, Rojo=Alerta, Azul=Neutral).
**Diseño Inclusivo:** Interfaz de alto contraste y fácil lectura, pensada para reducir la carga cognitiva.
**Privacidad:** Procesamiento local para proteger la identidad del usuario.

## 🛠️ Tecnologías Usadas

* **Lenguaje:** Python 3.10
**IA / Visión por Computador:** OpenCV, DeepFace, TensorFlow/Keras.
* **Interfaz Gráfica:** CustomTkinter (Modern UI).
**Hardware Soportado:** PC con webcam (escalable a Raspberry Pi/Wearables).

## Instalación

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/tu-usuario/ATFER.git](https://github.com/tu-usuario/ATFER.git)
    cd ATFER
    ```

2.  Crea un entorno virtual (Recomendado Python 3.10):
    ```bash
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    ```

3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

4.  Ejecuta la aplicación:
    ```bash
    python atfer_app.py
    ```

## Equipo PHYCOM

**Daniel Galarza** - Economía
**Symond Salazar** - Computación
**María Paula Chávez** - Telemática

---
*Este proyecto busca convertir la tecnología en un "héroe cotidiano" para quienes enfrentan barreras en la comunicación emocional.*
