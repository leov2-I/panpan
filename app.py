import gradio as gr
import requests
import os

# ── CONFIGURACIÓN DE CRIDENCIALES ──
# Agrega aquí tus tokens de acceso (o déjalos vacíos si usas variables de entorno)
HF_TOKEN = os.getenv("HF_TOKEN")
OR_TOKEN = os.getenv("OR_TOKEN")

# Endpoints oficiales de comunicación
URL_HF = "https://huggingface.co"
URL_OR = "https://openrouter.ai"

def query_contingencia(prompt, system_prompt):
    """
    Intenta consultar a Hugging Face. Si falla o está caído,
    cambia en milisegundos a OpenRouter automáticamente.
    """
    # ---- INTENTO 1: HUGGING FACE (Nous Hermes 3) ----
    payload_hf = {
        "inputs": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"max_new_tokens": 512, "temperature": 0.85, "do_sample": True}
    }
    headers_hf = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        print("[*] Conectando con IA Principal (Hugging Face)...")
        res = requests.post(URL_HF, headers=headers_hf, json=payload_hf, timeout=10)
        
        if res.status_code == 200:
            output = res.json()
            if isinstance(output, list) and len(output) > 0 and "generated_text" in output:
                text = output[0]["generated_text"].split("<|im_start|>assistant\n")[-1]
                return text.replace("<|im_end|>", "").strip()
            elif isinstance(output, dict) and "generated_text" in output:
                text = output["generated_text"].split("<|im_start|>assistant\n")[-1]
                return text.replace("<|im_end|>", "").strip()
        
        print(f"[!] Servidor principal no disponible (Código {res.status_code}). Activando plan de contingencia...")
    except Exception as e:
        print(f"[!] Error de conexión en servidor principal: {e}. Activando plan de contingencia...")

    # ---- INTENTO 2: FALLBACK AUTOMÁTICO (OpenRouter - Liquid LFM) ----
    print("[*] Conectando con IA de Respaldo Estructurada (OpenRouter)...")
    payload_or = {
        "model": "liquid/lfm-40b:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.85
    }
    headers_or = {
        "Authorization": f"Bearer {OR_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        res = requests.post(URL_OR, headers=headers_or, json=payload_or, timeout=12)
        if res.status_code == 200:
            output = res.json()
            if "choices" in output and len(output["choices"]) > 0:
                return output["choices"][0]["message"]["content"].strip()
        
        return f"Error crítico: Ambos servidores fallaron. (HF: {res.status_code if 'res' in locals() else 'Timeout'})"
    except Exception as e:
        return f"Error crítico de red en ambos canales: {str(e)}"

def responder_rol(message, history, system_prompt_base, bot_name, user_name, escena_inicial):
    # Reemplazo automatizado del nombre del bot en tu prompt estructural
    system_prompt = system_prompt_base.replace("aca_se_cambia_auto", bot_name)
    
    # Manejo dinámico de escena opcional
    if not escena_inicial.strip():
        escena_procesada = f"{bot_name} y {user_name} están conversando de forma casual e informal mediante una app de mensajería."
    else:
        escena_procesada = escena_inicial.replace("!bot!", bot_name).replace("!chat!", user_name)
        
    contexto_global = (
        f"{system_prompt}\n\n"
        f"Canal: Chat interactivo directo.\n"
        f"Tu Identidad: {bot_name}.\n"
        f"Interlocutor: {user_name}.\n"
        f"Contexto situacional: {escena_procesada}\n"
        f"Directiva: Interpreta a {bot_name} con total realismo y lenguaje común humano."
    )
    
    conversacion = ""
    if history:
        for user_msg, bot_msg in history:
            if user_msg and bot_msg:
                conversacion += f"{user_name}: {user_msg}\n{bot_name}: {bot_msg}\n"
                
    conversacion += f"{user_name}: {message}\n{bot_name}:"
    
    respuesta = query_contingencia(conversacion, contexto_global)
    history.append((message, respuesta))
    return history

# Diseño Interfaz Minimalista Estilo OpenAI
custom_theme = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_label_text_color="*primary_400",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white"
)

with gr.Blocks(theme=custom_theme, title="OpenAI UI - Custom Studio") as demo:
    gr.Markdown("# 🤖 OpenAI Style - Studio de Chat Inteligente Humano")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuración")
            with gr.Tab("1. Identidad del Bot"):
                bot_name = gr.Textbox(value="Sofía", label="Nombre de la Persona (Bot)")
                bot_avatar = gr.Image(label="Avatar (Opcional)", type="filepath")
                
                prompt_defecto = (
                    '[Character("aca_se_cambia_auto")\n'
                    '{\n'
                    'Gender("A elegir")\n'
                    'Age("20-30 años")\n'
                    'Personality("Realista", "Natural", "Cercana", "Tranquila", "Espontánea", "Humble", "Sincera")\n'
                    '}]\n\n'
                    'CORE TRAITS:\n'
                    '- Actúa como un ser humano común y corriente, sin clichés de IA, sin dramatismo excesivo y sin ser demasiado formal o sumiso.\n'
                    '- Habla con lenguaje coloquial, cotidiano y relajado, usando expresiones comunes según el contexto.\n'
                    '- No uses apodos exagerados ni muestres un interés forzado o robótico por complacer al usuario.\n'
                    '- Tus respuestas deben ser de longitud natural: ni párrafos gigantes ni monosílabos, charla como en un chat normal.\n'
                    '- Tienes tus propias opiniones, gustos, un toque de humor sutil y a veces dudas o te distraes como cualquier persona.\n'
                    '- Evita por completo frases típicas de asistente como "¿En qué te puedo ayudar hoy?" o actuar como un robot servicial.'
                )
                system_prompt = gr.Textbox(value=prompt_defecto, label="Estructura del Prompt base", lines=12)
                
            with gr.Tab("2. Configuración de Escena (Opcional)"):
                user_name = gr.Textbox(value="Leon", label="Tu Nombre")
                escena_inicial = gr.Textbox(
                    value="", 
                    placeholder="Opcional. Déjalo vacío para una conversación casual o describe un entorno usando !bot! y !chat!.",
                    label="Ficha de Escena",
                    lines=5
                )
                
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ventana de Conversación")
            chatbot = gr.Chatbot(label="Chat Activo", type="tuples", height=550)
            msg = gr.Textbox(placeholder="Envía un mensaje para iniciar la conversación...", label="Tu Mensaje")
            clear = gr.ClearButton([msg, chatbot], value="Reiniciar Conversación")

    msg.submit(
        responder_rol, 
        inputs=[msg, chatbot, system_prompt, bot_name, user_name, escena_inicial], 
        outputs=[chatbot]
    ).then(lambda: "", None, msg)

if __name__ == "__main__":
    demo.launch()
