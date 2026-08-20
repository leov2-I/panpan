import os
import gradio as gr
import requests
from huggingface_hub import InferenceClient

# Inicialización del cliente de Hugging Face usando variables de entorno
# Asegúrate de configurar 'HF_TOKEN' en la sección Environment de Render si usas modelos protegidos
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

def responder_chat(mensaje, historial):
    """
    Función encargada de procesar la entrada de texto y comunicarse con Hugging Face.
    Modifica el modelo según tus necesidades.
    """
    try:
        # Ejemplo con un modelo conversacional popular
        modelo = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Estructurar el historial de forma nativa para las API de Hugging Face
        mensajes_api = []
        for usuario, asistente in historial:
            if usuario:
                mensajes_api.append({"role": "user", "content": usuario})
            if asistente:
                mensajes_api.append({"role": "assistant", "content": asistente})
                
        mensajes_api.append({"role": "user", "content": mensaje})
        
        respuesta_completa = ""
        for token in client.chat_completion(
            model=modelo,
            messages=mensajes_api,
            max_tokens=500,
            stream=True
        ):
            token_texto = token.choices[0].delta.content
            if token_texto:
                respuesta_completa += token_texto
                yield respuesta_completa
                
    except Exception as e:
        yield f"⚠️ Ocurrió un error al procesar el mensaje: {str(e)}"

# --- ESTRUCTURA DE LA INTERFAZ DE GRADIO (CORREGIDA) ---
with gr.Blocks(theme="ocean", title="OpenAI UI - Custom Studio") as demo:
    gr.Markdown("# 🤖 Mi Asistente Personal")
    
    with gr.Row():
        # Ajustes generales en la columna lateral izquierda
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Ajustes del Sistema")
            gr.Markdown("Interfaz optimizada y lista para producción en Render.")
            
        # Ventana de conversación en la columna derecha
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ventana de Conversación")
            
            # CORRECCIÓN 1: Se remueve 'type="tuples"' para evitar incompatibilidad de versión
            chatbot = gr.Chatbot(label="Chat Activo", height=550)
            
            msg = gr.Textbox(
                placeholder="Envía un mensaje para iniciar la conversación...", 
                label="Tu Mensaje"
            )
            clear = gr.ClearButton([msg, chatbot], value="Reiniciar Conversación")

    # Envío de datos al presionar Enter o hacer submit en el cuadro de texto
    msg.submit(responder_chat, inputs=[msg, chatbot], outputs=[chatbot]).then(
        lambda: gr.update(value=""), None, [msg]
    )

# Lanzamiento del servidor escuchando los parámetros obligatorios de Render
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
