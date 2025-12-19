import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Environment Variables
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # <- en Render
ASFIES_TOKEN = os.getenv("ASFIES_TOKEN")       # <- en Render (ej. ASFIES150320)

# OpenAI Client (solo si hay key)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Nombre del archivo de la matriz (debe estar en la raíz del repo)
MATRIZ_FILE = "matriz.xlsx"


# =========================
# Helpers
# =========================
def obtener_recomendaciones(perfil: dict):
    """
    Filtra la matriz por rango de ventas y antigüedad y devuelve top 3.
    """
    try:
        if not os.path.exists(MATRIZ_FILE):
            print(f"[ERROR] No se encontró el archivo: {MATRIZ_FILE}")
            return None

        df = pd.read_excel(MATRIZ_FILE)

        # 1) Filtro por Rango de Ventas
        # El PHP enviará el rango exacto, ej: "1,000,001 - 10,000,000"
        rango = (perfil.get("ventas_rango") or "").strip()
        if rango and rango in df.columns:
            # Filtramos solo las financieras que tienen 'X' en ese rango
            df = df[df[rango].astype(str).str.upper().str.strip() == "X"]

        # 2) Filtro por Antigüedad (mapeo conservador)
        antiguedad_map = {
            "Menos de 1 año": 0,
            "Entre 1 y 3 años": 1,
            "Entre 3 y 5 años": 3,
            "Más de 5 años": 5
        }
        user_anios = antiguedad_map.get((perfil.get("antiguedad") or "").strip(), 0)

        # Columna esperada: 'Antigúedad'
        if "Antigúedad" in df.columns:
            df["min_anios"] = (
                df["Antigúedad"]
                .astype(str)
                .str.extract(r"(\d+)")
                .fillna(0)
                .astype(int)
            )
            df = df[df["min_anios"] <= user_anios]
        else:
            # Si la columna no existe, no truena; solo avisa.
            print("[WARN] La columna 'Antigúedad' no existe en la matriz. Se omite filtro de antigüedad.")

        # 3) Selección de las mejores 3 opciones
        top_df = df.head(3).copy()
        if top_df.empty:
            return None

        recomendaciones = []
        for _, row in top_df.iterrows():
            recomendaciones.append({
                "financiera": row.get("Financiera", ""),
                "tipo": row.get("Tipo de financiamiento", ""),
                "ventaja": row.get("Como ayuda este financiamiento", ""),
                "monto": row.get("Montos en pesos", ""),
                "plazo": row.get("Plazos", "")
            })

        return recomendaciones

    except Exception as e:
        print(f"[ERROR] Procesando matriz: {e}")
        return None


def generar_diagnostico_gpt(perfil: dict, recomendaciones: list):
    """
    Genera texto consultivo (máx 2 párrafos) con GPT.
    Si no hay API key o falla, regresa un fallback profesional.
    """
    fallback = (
        "Nuestro equipo de analistas procesará su información detalladamente "
        "para entregarle el reporte final vía correo electrónico."
    )

    try:
        if client is None:
            print("[WARN] OPENAI_API_KEY no configurada. Se devuelve fallback.")
            return fallback

        tiene_garantia = bool(perfil.get("tiene_garantia_inmueble", False))
        contexto_garantia = (
            "con garantía inmobiliaria bajo la metodología CAOM"
            if tiene_garantia else
            "sin garantía inmobiliaria"
        )

        prompt = f"""
Eres un asesor financiero experto de ASFIES Negocios Consulting.
Analiza el siguiente perfil empresarial y las opciones encontradas:

CLIENTE: {perfil.get('nombre_empresa')}
ACTIVIDAD: {perfil.get('actividad_economica')}
VENTAS: {perfil.get('ventas_rango')}
ESTADO: Financiamiento {contexto_garantia}.

OPCIONES ENCONTRADAS (JSON):
{recomendaciones}

TAREA:
Escribe un diagnóstico profesional de máximo 2 párrafos.
Explica por qué estas opciones son estratégicas para su situación.
Usa un tono consultivo, elegante y técnico.
Si tiene garantía, menciona que se aplicará la 'Arquitectura y Optimización de Capital (CAOM)'.
        """.strip()

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] GPT diagnostico: {e}")
        return fallback


# =========================
# Routes
# =========================
@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint simple para verificar que el servicio está vivo.
    """
    return jsonify({
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "matriz_exists": os.path.exists(MATRIZ_FILE)
    })


@app.route("/diagnostico", methods=["POST"])
def handle_diagnostico():
    # Validación de Token de Seguridad
    auth_header = request.headers.get("Authorization")
    if not ASFIES_TOKEN:
        return jsonify({"error": "Server token not configured (ASFIES_TOKEN missing)"}), 500

    if auth_header != f"Bearer {ASFIES_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    datos = request.json or {}

    # 1) Obtener datos de la matriz
    opciones = obtener_recomendaciones(datos)
    if not opciones:
        return jsonify({
            "status": "not_found",
            "mensaje": "No se encontraron financieras que coincidan exactamente con el perfil, pero un consultor revisará su caso manualmente."
        })

    # 2) Generar texto con IA
    diagnostico_ia = generar_diagnostico_gpt(datos, opciones)

    # 3) Respuesta Final
    return jsonify({
        "status": "success",
        "header": "Tenemos las siguientes estrategias" if datos.get("tiene_garantia_inmueble") else "Opciones de Financiamiento Identificadas",
        "diagnostico_ia": diagnostico_ia,
        "recomendaciones": opciones
    })


# =========================
# Main
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
