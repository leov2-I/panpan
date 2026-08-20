import os
import gradio as gr
import requests
from huggingface_hub import InferenceClient

# Inicialización del cliente de Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

def responder_chat(mensaje, historial):
    try:
        modelo = "meta-llama/Llama-3.2-3B-Instruct"
        mensajes_api = []
        
        # Procesar el historial de forma compatible con Gradio 5
        for turno in historial:
            if isinstance(turno, dict):
                usuario = turno.get("text", "") if turno.get("role") == "user" else ""
                asistente = turno.get("text", "") if turno.get("role") == "assistant" else ""
            else:
                try:
                    usuario, asistente = turno, turno
                except Exception:
                    usuario, asistente = turno, ""
                
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
            token_texto = token.choices.delta.content
            if token_texto:
                respuesta_completa += token_texto
                yield respuesta_completa
                
    except Exception as e:
        yield f"⚠️ Error de conexión: {str(e)}"

# Interfaz gráfica original (Gradio Blocks con tema Ocean)
with gr.Blocks(theme="ocean", title="OpenAI UI - Custom Studio") as demo:
    gr.Markdown("# 🤖 Mi Asistente Personal")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Ajustes del Sistema")
            gr.Markdown("Interfaz visual completamente optimizada.")
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ventana de Conversación")
            chatbot = gr.Chatbot(label="Chat Activo", height=550)
            msg = gr.Textbox(
                placeholder="Envía un mensaje para iniciar la conversación...", 
                label="Tu Mensaje"
            )
            clear = gr.ClearButton([msg, chatbot], value="Reiniciar Conversación")

    # Manejo de envío nativo para Gradio 5 (evita el bloqueo de localhost)
    msg.submit(responder_chat, inputs=[msg, chatbot], outputs=[chatbot])
    msg.submit(lambda: "", None, [msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000, share=False)
