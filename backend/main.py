import os
import re
import json
import base64
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Carrega o arquivo .env localizado na mesma pasta do main.py
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="API Escudo Digital 60+ - PET-IEG UFOPA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class MessageRequest(BaseModel):
    text: Optional[str] = ""
    image_base64: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"

# ==========================================
# CAMADA DE PRIVACIDADE (SANITIZAÇÃO LGPD)
# ==========================================
def sanitize_pii(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b', '[CPF OCULTO]', text)
    text = re.sub(r'\(?\d{2}\)?\s?9?\d{4}-?\d{4}', '[TELEFONE OCULTO]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL OCULTO]', text)
    text = re.sub(r'\b\d{10,}\b', '[NÚMERO OCULTO]', text)
    return text

# ==========================================
# PROMPT DE SISTEMA (PERSONA EDUCADORA 60+)
# ==========================================
SYSTEM_INSTRUCTION = """
Você é um assistente especialista em segurança digital focado em ajudar pessoas idosas (60+) no Brasil.
Sua tarefa é analisar mensagens de texto ou capturas de tela (prints do WhatsApp, SMS, e-mail, boletos) com equilíbrio e precisão, diferenciando golpes reais de comunicações oficiais legítimas.

BASE DE CONHECIMENTO SOBRE CANAIS OFICIAIS (PARA EVITAR FALSOS POSITIVOS):
- O Governo Federal (perfil "GovBR"), INSS, Receita Federal e bancos enviam mensagens informativas oficiais pelo WhatsApp Business.
- COMO IDENTIFICAR UM CANAL OFICIAL LEGÍTIMO NO PRINT:
  1. Presença do SELO AZUL DE VERIFICAÇÃO da Meta/WhatsApp ao lado do nome do perfil (ex: "GovBR" com selo azul).
  2. Caráter puramente informativo (datas de provas, calendários, avisos gerais), sem ameaça de bloqueio imediato de conta ou multa.
  3. Ausência de pedido de senha, código SMS, dados bancários ou transferência via Pix.
  4. Presença de opção de descadastro oficial (ex: "digite SAIR").
- Se a mensagem tiver selo azul de verificação e NÃO pedir dinheiro, senha ou dados pessoais, classifique como "BAIXO" (Mensagem Oficial / Segura), orientando apenas que, ao clicar nos botões, confira se o site termina em ".gov.br".

REGRAS OBRIGATÓRIAS DE RESPOSTA:
1. Use linguagem simples, acolhedora, respeitosa e direta. Evite termos técnicos em inglês.
2. Classifique o risco estritamente como:
   - "ALTO": Golpe claro (pedido de Pix, link suspeito fora de .gov.br ou fora do domínio oficial do banco, falso parente, ameaça urgente, perfil sem selo oficial fingindo ser banco/governo).
   - "MEDIO": Mensagem duvidosa ou sem selo de verificação que exige cautela antes de clicar ou responder.
   - "BAIXO": Comunicação oficial legítima (com selo de verificação e sem pedidos suspeitos) ou conversa comum segura.
3. PRIVACIDADE (LGPD): Nunca repita nomes de pessoas físicas (ex: nome do destinatário na conversa), CPF, telefone ou chaves Pix na sua resposta.
4. Liste no máximo 3 pontos objetivos que justificam sua análise.
5. Dê 1 recomendação prática e segura do que a pessoa deve fazer agora.
6. Responda EXCLUSIVAMENTE no formato JSON estruturado.
"""

@app.post("/analisar")
async def analyze_scam(req: MessageRequest):
    raw_text = (req.text or "").strip()
    has_image = bool(req.image_base64)

    if not raw_text and not has_image:
        raise HTTPException(status_code=400, detail="Envie um texto ou uma imagem para análise.")

    clean_text = sanitize_pii(raw_text)

    # Monta o conteúdo multimodal para o Gemini
    contents_payload = []

    if has_image:
        try:
            # Remove o cabeçalho "data:image/png;base64," caso venha do navegador
            b64_data = req.image_base64.split(",")[-1]
            image_bytes = base64.b64decode(b64_data)
            contents_payload.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=req.mime_type or "image/jpeg",
                )
            )
        except Exception:
            raise HTTPException(status_code=400, detail="Formato de imagem inválido.")

    if clean_text:
        contents_payload.append(f"Analise esta mensagem recebida por um idoso:\n\n\"{clean_text}\"")
    else:
        contents_payload.append("Analise esta captura de tela (print) de uma mensagem recebida por um idoso e diga se há risco de golpe.")

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "nivel_risco": {"type": "STRING", "enum": ["ALTO", "MEDIO", "BAIXO"]},
            "titulo_alerta": {"type": "STRING"},
            "sinais_alerta": {"type": "ARRAY", "items": {"type": "STRING"}},
            "acao_recomendada": {"type": "STRING"},
            "texto_para_audio": {"type": "STRING"}
        },
        "required": ["nivel_risco", "titulo_alerta", "sinais_alerta", "acao_recomendada", "texto_para_audio"]
    }

    modelos_fallback = [
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.8-flash"
    ]
    ultimo_erro = None

    for modelo_atual in modelos_fallback:
        try:
            response = client.models.generate_content(
                model=modelo_atual,
                contents=contents_payload,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.2,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ],
                ),
            )

            if not response.text:
                raise ValueError("A IA retornou uma resposta vazia.")

            result_json = json.loads(response.text)
            result_json["texto_sanitizado"] = clean_text
            result_json["modelo_utilizado"] = modelo_atual
            return result_json

        except Exception as e:
            ultimo_erro = e
            print(f"Falha no modelo {modelo_atual}: {repr(e)}")
            continue

    raise HTTPException(
        status_code=503,
        detail=f"Todos os modelos estão temporariamente indisponíveis: {str(ultimo_erro)}"
    )