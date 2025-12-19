import os
import smtplib
import pandas as pd
from email.message import EmailMessage
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# =========================
# Flask App Configuration
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Environment Variables
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASFIES_TOKEN = os.getenv("ASFIES_TOKEN")

# Inicialización cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Nombre del archivo matriz (Asegúrate que sea minúscula o coincida con tu repo)
MATRIZ_FILE = "matriz.xlsx"

# SMTP / Email Configuration (mihosting / Webempresa)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER or "")
MAIL_TO = os.getenv("MAIL_TO", "contacto@asfies.mx")

MAIL_USE_TLS = (os.getenv("MAIL_USE_TLS", "true").lower() == "true")
MAIL_USE_SSL = (os.getenv("MAIL_USE_SSL", "false").lower() == "true")

# =========================
# Lógica de Matriz
# =========================
def obtener_recomendaciones(perfil: dict):
    try:
        if not os.path.exists(MATRIZ_FILE):
            print(f"[ERROR] No se encontró el archivo: {MATRIZ_FILE}")
            return None

        df = pd.read_excel(MATRIZ_FILE)

        # 1) Filtro por Rango de Ventas
        rango = (perfil.get("ventas_rango") or "").strip()
        if rango and rango in df.columns:
            df = df[df[rango].astype(str).str.upper().str.strip() == "X"]

        # 2) Filtro por Antigüedad
        antiguedad_map = {
            "Menos de 1 año": 0,
            "Entre 1 y 3 años": 1,
            "Entre 3 y 5 años": 3,
            "Más de 5 años": 5
        }
        user_anios = antiguedad_map.get((perfil.get("antiguedad") or "").strip(), 0)

        if "Antigúedad" in df.columns:
            df["min_anios"] = (
                df["Antigúedad"].astype(str)
                .str.extract(r"(\d+)")
                .fillna(0)
                .astype(int)
            )
            df = df[df["min_anios"] <= user_anios]

        top_df = df.head(3).copy()
        if top_df.empty:
            return None

        recomendaciones = []
        for _, row in top_df.iterrows():
            recomendaciones.append({
                "financiera": str(row.get("Financiera", "")),
                "tipo": str(row.get("Tipo de financiamiento", "")),
                "ventaja": str(row.get("Como ayuda este financiamiento", "")),
                "monto": str(row.get("Montos en pesos", "")),
                "plazo": str(row.get("Plazos", ""))
            })
        return recomendaciones

    except Exception as e:
        print(f"[ERROR] Procesando matriz: {e}")
        return None

# =========================
# Lógica de IA (GPT)
# =========================
def generar_diagnostico_gpt(perfil: dict, recomendaciones: list):
    fallback = "Nuestro equipo de analistas procesará su información para entregarle el reporte final vía email."
    try:
        if client is None: return fallback

        tiene_garantia = bool(perfil.get("tiene_garantia_inmueble", False))
        contexto = "con garantía inmobiliaria (CAOM)" if tiene_garantia else "sin garantía inmobiliaria"

        prompt = f"""
        Eres un asesor experto de ASFIES Negocios Consulting.
        Analiza: Cliente {perfil.get('nombre_empresa')} ({perfil.get('actividad_economica')}).
        Ventas: {perfil.get('ventas_rango')}. Estado: {contexto}.
        Opciones Matriz: {recomendaciones}

        TAREA: Escribe un diagnóstico de 2 párrafos, elegante y técnico. 
        Si tiene garantía, menciona la 'Arquitectura y Optimización de Capital (CAOM)'.
        """.strip()

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT: {e}")
        return fallback

# =========================
# Notificaciones SMTP
# =========================
def enviar_correo_lead(perfil: dict, recomendaciones: list, diagnostico: str):
    tiene_garantia = bool(perfil.get("tiene_garantia_inmueble", False))
    solicita_contacto = str(perfil.get("solicita_contacto", "")).lower() in ["si", "sí", "true", "1", "a"]

    if not (tiene_garantia or solicita_contacto):
        return False

    msg = EmailMessage()
    msg["Subject"] = f"NUEVO LEAD CAOM - {perfil.get('nombre_empresa','S/E')}"
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO

    body = f"""LEAD - DIAGNÓSTICO ASFIES
    
CONTACTO: {perfil.get('nombre','')} {perfil.get('apellido','')}
Empresa: {perfil.get('nombre_empresa','')}
Tel/Email: {perfil.get('telefono','')} / {perfil.get('correo','')}

GARANTÍA INMOBILIARIA: {'SÍ' if tiene_garantia else 'NO'}
Inmueble: {perfil.get('inmueble_tipo','')} | Valor: {perfil.get('inmueble_valor_aprox','')}
Ubicación: {perfil.get('ubicacion_google_maps','')}

DIAGNÓSTICO IA:
{diagnostico}

RECOMENDACIONES:
{recomendaciones}
    """
    msg.set_content(body)

    try:
        if MAIL_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.ehlo()
                if MAIL_USE_TLS: server.starttls(); server.ehlo()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        return True
    except Exception as e:
        print(f"[ERROR] SMTP: {e}")
        return False

# =========================
# Rutas
# =========================
@app.route("/diagnostico", methods=["POST"])
def handle_diagnostico():
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {ASFIES_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    datos = request.json or {}
    opciones = obtener_recomendaciones(datos)
    
    if not opciones:
        return jsonify({"status": "not_found", "mensaje": "No se hallaron coincidencias."})

    diag = generar_diagnostico_gpt(datos, opciones)
    email_ok = enviar_correo_lead(datos, opciones, diag)

    return jsonify({
        "status": "success",
        "header": "Tenemos las siguientes estrategias" if datos.get("tiene_garantia_inmueble") else "Opciones Identificadas",
        "diagnostico_ia": diag,
        "recomendaciones": opciones,
        "email_enviado": email_ok
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))