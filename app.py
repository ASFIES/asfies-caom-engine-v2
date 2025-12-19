import os
import smtplib
import pandas as pd
from email.message import EmailMessage
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Configuración desde Variables de Entorno en Render
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASFIES_TOKEN = os.getenv("ASFIES_TOKEN")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MATRIZ_FILE = "matriz.xlsx"

# Configuración SMTP
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER or "")
MAIL_TO = os.getenv("MAIL_TO", "contacto@asfies.mx")

def filtrar_matriz(perfil: dict):
    try:
        if not os.path.exists(MATRIZ_FILE): return None
        df = pd.read_excel(MATRIZ_FILE)

        # 1. Filtro por Ventas
        rango_v = perfil.get("ventas_rango", "")
        if rango_v in df.columns:
            df = df[df[rango_v].astype(str).str.upper() == "X"]

        # 2. Análisis de Monto y Plazo (Búsqueda por palabras clave o coincidencia parcial)
        monto_u = str(perfil.get("monto_requerido", ""))
        plazo_u = str(perfil.get("plazo_requerido", ""))
        
        # Priorizamos filas que mencionen montos o plazos similares en sus columnas de texto
        if monto_u:
            df['match_monto'] = df['Montos en pesos'].astype(str).str.contains(monto_u.split(' ')[0], case=False)
        if plazo_u:
            df['match_plazo'] = df['Plazos'].astype(str).str.contains(plazo_u.split(' ')[0], case=False)

        # 3. Filtro por Antigüedad
        ant_map = {"Menos de 1 año": 0, "Entre 1 y 3 años": 1, "Entre 3 y 5 años": 3, "Más de 5 años": 5}
        user_anios = ant_map.get(perfil.get("antiguedad", ""), 0)
        df['min_anios'] = df['Antigúedad'].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
        df = df[df['min_anios'] <= user_anios]

        top_df = df.head(3).copy()
        if top_df.empty: return None

        tiene_garantia = bool(perfil.get("tiene_garantia_inmueble", False))
        recomendaciones = []
        
        for _, row in top_df.iterrows():
            # REGLA: Si hay garantía, se oculta el nombre de la financiera
            nombre_display = "Estrategia Recomendada" if tiene_garantia else row.get("Financiera", "Opción Identificada")
            
            recomendaciones.append({
                "financiera": nombre_display,
                "tipo": str(row.get("Tipo de financiamiento", "")),
                "caracteristicas": str(row.get("Como ayuda este financiamiento", "")),
                "monto_ref": str(row.get("Montos en pesos", "")),
                "plazo_ref": str(row.get("Plazos", ""))
            })
        return recomendaciones
    except Exception as e:
        print(f"Error Matriz: {e}")
        return None

def enviar_email_lead(perfil: dict, diag: str):
    if not bool(perfil.get("tiene_garantia_inmueble", False)) and not perfil.get("solicita_contacto"):
        return False
    
    msg = EmailMessage()
    msg["Subject"] = f"NUEVO LEAD CAOM - {perfil.get('nombre_empresa','S/E')}"
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    
    content = f"""DETALLE DEL LEAD ASFIES CAOM
    
CONTACTO: {perfil.get('nombre')} {perfil.get('apellido')}
Empresa: {perfil.get('nombre_empresa')}
Tel: {perfil.get('telefono')} | Email: {perfil.get('correo')}

SOLICITUD:
Monto: {perfil.get('monto_requerido')}
Plazo: {perfil.get('plazo_requerido')}
Ventas: {perfil.get('ventas_rango')}

DATOS PATRIMONIALES (GARANTÍA):
Tipo: {perfil.get('inmueble_tipo')}
Valor: {perfil.get('inmueble_valor_aprox')}
¿Hipotecado?: {perfil.get('esta_hipotecado')}
Maps: {perfil.get('ubicacion_google_maps')}

DIAGNÓSTICO IA:
{diag}
"""
    msg.set_content(content)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except: return False

@app.route("/diagnostico", methods=["POST"])
def handle_diagnostico():
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {ASFIES_TOKEN}": return jsonify({"error": "Unauthorized"}), 401
    
    datos = request.json
    opciones = filtrar_matriz(datos)
    
    # Generar diagnóstico con OpenAI (usando gpt-4o-mini para velocidad)
    contexto = "con respaldo patrimonial CAOM" if datos.get("tiene_garantia_inmueble") else "financiamiento tradicional"
    prompt = f"Genera un diagnóstico financiero para {datos.get('nombre_empresa')}. Monto: {datos.get('monto_requerido')}. Situación: {contexto}. Opciones: {opciones}. Sé técnico y elegante."
    
    diag_ia = "Procesando diagnóstico detallado..."
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        diag_ia = resp.choices[0].message.content
    except: pass

    enviar_email_lead(datos, diag_ia)

    return jsonify({
        "status": "success",
        "header": "Estrategias Recomendadas" if datos.get("tiene_garantia_inmueble") else "Opciones de Financiamiento",
        "diagnostico_ia": diag_ia,
        "recomendaciones": opciones
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))