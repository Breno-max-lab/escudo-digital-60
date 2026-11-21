import os
import re
import io
import json
import base64
import edge_tts
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("A variável GEMINI_API_KEY não foi encontrada no ambiente.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Escudo Digital 60+ API",
    description="Backend multimodal com proteção LGPD e síntese de voz neural para inclusão digital de idosos.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 1. SANITIZAÇÃO LGPD (ANONIMIZAÇÃO DE DADOS SENSÍVEIS)
# ==============================================================================
def sanitize_pii(text: str) -> str:
    if not text:
        return ""
    # Mascara CPFs (000.000.000-00 ou 11 dígitos seguidos)
    text = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF_OCULTO]", text)
    # Mascara telefones brasileiros
    text = re.sub(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\d{4}|\d{4})-?\d{4}\b", "[TELEFONE_OCULTO]", text)
    # Mascara e-mails
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL_OCULTO]", text)
    # Mascara possíveis números de cartão/conta longos
    text = re.sub(r"\b\d{13,19}\b", "[NUMERO_CONTA_OCULTO]", text)
    return text


# ==============================================================================
# 2. MODELOS DE DADOS (REQUEST / RESPONSE)
# ==============================================================================
class AnalyzeRequest(BaseModel):
    text: Optional[str] = ""
    image_base64: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"


class AnalyzeResponse(BaseModel):
    nivel_risco: str  # "ALTO", "MEDIO" ou "BAIXO"
    titulo_alerta: str
    sinais_alerta: List[str]
    acao_recomendada: str
    texto_para_audio: str


class SpeechRequest(BaseModel):
    text: str


# ==============================================================================
# 3. PROMPT CALIBRADO PARA O PÚBLICO 60+
# ==============================================================================
SYSTEM_PROMPT = """
Você é o assistente de segurança digital do projeto 'Escudo Digital 60+' (PET-IEG / UFOPA, Santarém - Pará).
Sua missão é analisar mensagens de texto ou capturas de tela (prints) de WhatsApp, SMS ou e-mail para ajudar pessoas idosas (60+) a identificarem golpes virtuais com clareza, acolhimento e linguagem simples.

REGRAS IMPORTANTES DE CALIBRAÇÃO:
1. Diferencie golpes reais de comunicações legítimas:
   - Se a imagem mostrar um perfil comercial oficial com selo de verificação azul/verde (como GovBR, Caixa, Banco do Brasil, INSS) apenas informando datas ou serviços sem pedir senha, código SMS ou Pix imediato, classifique como risco "BAIXO" (seguro/informativo), orientando sempre a conferir se eventuais sites terminam em '.gov.br' ou abrir o app oficial diretamente.
   - Classifique como "ALTO" quando houver pedido urgente de Pix/dinheiro para parente ou terceiro, links suspeitos que imitam bancos/governo (phishing), solicitação de senha/código de verificação, instalação de aplicativos externos ou cobrança de taxa para liberar prêmio/benefício.
   - Classifique como "MEDIO" quando a mensagem for ambígua, de remetente desconhecido ou contiver ofertas exageradas sem pedido direto de dinheiro.
2. Linguagem acessível:
   - Use frases curtas, diretas e sem jargões técnicos em inglês (evite palavras como 'phishing', 'malware' ou 'spoofing'; explique o risco em português claro).
   - O campo 'texto_para_audio' deve ser um resumo fluido, natural e acolhedor para ser lido em voz alta para um idoso.

Responda EXCLUSIVAMENTE em JSON válido seguindo exatamente esta estrutura:
{
  "nivel_risco": "ALTO" | "MEDIO" | "BAIXO",
  "titulo_alerta": "Título curto e claro do resultado",
  "sinais_alerta": ["Ponto 1 observado na mensagem", "Ponto 2 observado"],
  "acao_recomendada": "Orientação prática do que o idoso deve fazer agora",
  "texto_para_audio": "Texto completo e natural para leitura em voz alta"
}
"""


# ==============================================================================
# 4. ENDPOINTS DA API
# ==============================================================================
@app.get("/")
def health_check():
    return {"status": "online", "service": "Escudo Digital 60+ API"}


@app.post("/analisar", response_model=AnalyzeResponse)
async def analisar_mensagem(req: AnalyzeRequest):
    clean_text = sanitize_pii(req.text or "")
    has_image = bool(req.image_base64)

    if not clean_text.strip() and not has_image:
        raise HTTPException(status_code=400, detail="Envie um texto ou uma imagem para análise.")

    contents = [SYSTEM_PROMPT]

    if clean_text.strip():
        contents.append(f"Texto informado pelo usuário:\n\"\"\"{clean_text}\"\"\"")

    if has_image:
        try:
            b64_data = req.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=req.mime_type or "image/jpeg"
                )
            )
            contents.append("Analise também todo o conteúdo visual, remetente, selos de verificação, links e textos presentes nesta captura de tela.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar imagem enviada: {str(e)}")

    # Fallback resiliente entre modelos Gemini
    candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_error = None

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            raw_json = response.text.strip()
            parsed = json.loads(raw_json)
            return AnalyzeResponse(**parsed)
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"Falha ao consultar modelos de IA: {str(last_error)}")


@app.post("/falar")
async def gerar_audio_neural(req: SpeechRequest):
    texto_limpo = req.text.strip()[:1500]
    if not texto_limpo:
        raise HTTPException(status_code=400, detail="Texto vazio para leitura.")

    try:
        communicate = edge_tts.Communicate(
            texto_limpo,
            voice="pt-BR-FranciscaNeural",
            rate="-5%"
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        return StreamingResponse(audio_buffer, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar áudio neural: {str(e)}")