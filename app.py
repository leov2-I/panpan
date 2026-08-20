import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from huggingface_hub import InferenceClient

# Inicialización segura de la API y el cliente HF
app = FastAPI(title="Panpan AI API")
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# Modelo de datos para recibir los mensajes estructurados
class ChatRequest(BaseModel):
    mensaje: str
    historial: List[Dict[str, str]] = []  # Lista de {"role": "user", "content": "..."}

@app.get("/")
def inicio():
    return {"estado": "online", "mensaje": "API de Panpan corriendo con éxito en Render"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        modelo = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Construir los mensajes para Hugging Face
        mensajes_api = []
        for turno in request.historial:
            mensajes_api.append({"role": turno.get("role", "user"), "content": turno.get("content", "")})
            
        mensajes_api.append({"role": "user", "content": request.mensaje})
        
        # Generador para hacer streaming de la respuesta
        def generar_respuesta():
            try:
                for token in client.chat_completion(
                    model=modelo,
                    messages=mensajes_api,
                    max_tokens=500,
                    stream=True
                ):
                    token_texto = token.choices.delta.content
                    if token_texto:
                        yield token_texto
            except Exception as stream_err:
                yield f"⚠️ Error en streaming: {str(stream_err)}"

        return StreamingResponse(generar_respuesta(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en el servidor: {str(e)}")

if __name__ == "__main__":
    # Render requiere obligatoriamente escuchar en el puerto 10000 e interfaz 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=10000)
