import os
import re
import json
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


app = Flask(__name__)
CORS(app)

# =========================
# ENV
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ASFIES_TOKEN = os.getenv("ASFIES_TOKEN", "").strip()

MODEL = os.getenv("MODEL", "gpt-4o-mini").strip()
MATRIZ_FILE = os.getenv("MATRIZ_FILE", "matriz.xlsx").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int((os.getenv("SMTP_PORT", "587") or "587").strip())
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "").strip()
SMTP_TO = os.getenv("SMTP_TO", "").strip()

_MATRIZ_CACHE: Optional[pd.DataFrame] = None


def _log(msg: str):
    print(f"[ASFIES-CAOM] {msg}", flush=True)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _as_bool(v: Any) -> bool:
    """
    Convierte True/False aunque llegue como string ("true", "false", "1", "0").
    """
    if isinstance(v, bool):
        return v
    s = _safe_str(v).lower()
    return s in ("true", "1", "yes", "si", "sí", "y")


def _load_matriz() -> pd.DataFrame:
    global _MATRIZ_CACHE
    if _MATRIZ_CACHE is not None:
        return _MATRIZ_CACHE

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, MATRIZ_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró '{MATRIZ_FILE}' en el repo. Ruta esperada: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    _MATRIZ_CACHE = df
    _log(f"Matriz cargada OK: {path} | filas={len(df)} cols={len(df.columns)}")
    return df


def _extract_min_years(val: Any) -> int:
    s = _safe_str(val)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _map_antiguedad_user(texto: str) -> int:
    mapa = {
        "Menos de 1 año": 0,
        "Entre 1 y 3 años": 1,
        "Entre 3 y 5 años": 3,
        "Más de 5 años": 5,
    }
    return mapa.get(texto, 0)


def obtener_recomendaciones_financieras(perfil: Dict[str, Any]) -> List[Dict[str, Any]]:
    df = _load_matriz().copy()

    rango = _safe_str(perfil.get("ventas_rango"))
    if rango and rango in df.columns:
        df = df[df[rango].astype(str).str.upper().str.strip() == "X"]

    user_years = _map_antiguedad_user(_safe_str(perfil.get("antiguedad")))

    ant_col = None
    for cand in ["Antigúedad", "Antiguedad", "Antigüedad"]:
        if cand in df.columns:
            ant_col = cand
            break

    if ant_col:
        df["min_anios"] = df[ant_col].apply(_extract_min_years)
        df = df[df["min_anios"] <= user_years]
    else:
        df["min_anios"] = 0

    tipo_user = _safe_str(perfil.get("tipo_financiamiento"))
    tipo_col = None
    for cand in ["Tipo de financiamiento", "Tipo", "Producto", "Tipo de financiamiento "]:
        if cand in df.columns:
            tipo_col = cand
            break
    if tipo_col and tipo_user:
        df = df[df[tipo_col].astype(str).str.contains(tipo_user, case=False, na=False)]

    top_df = df.head(3).copy()
    if top_df.empty:
        return []

    def col(*names: str) -> str:
        for n in names:
            if n in top_df.columns:
                return n
        return ""

    c_fin = col("Financiera", "Nombre", "Institución")
    c_tipo = col("Tipo de financiamiento", "Tipo", "Producto")
    c_ayu = col("Como ayuda este financiamiento", "Cómo ayuda este financiamiento", "Ayuda", "Descripción")
    c_mto = col("Montos en pesos", "Monto", "Montos")
    c_plz = col("Plazos", "Plazo")

    recs: List[Dict[str, Any]] = []
    for _, row in top_df.iterrows():
        recs.append({
            "financiera": _safe_str(row.get(c_fin)),
            "tipo": _safe_str(row.get(c_tipo)),
            "caracteristicas": _safe_str(row.get(c_ayu)),
            "monto_ref": _safe_str(row.get(c_mto)),
            "plazo_ref": _safe_str(row.get(c_plz)),
        })
    return recs


def recomendaciones_a_estrategias(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Versión simple: 3 estrategias genéricas sin exponer institución
    nombres = ["Estrategia Alfa", "Estrategia Beta", "Estrategia Delta"]
    estrategias = []
    for i, r in enumerate(recs, start=1):
        estrategias.append({
            "estrategia": f"{nombres[i-1] if i-1 < len(nombres) else f'Estrategia {i}'}: Optimización con Respaldo Inmobiliario",
            "tipo": _safe_str(r.get("tipo")),
            "caracteristicas": _safe_str(r.get("caracteristicas")),
            "monto_ref": _safe_str(r.get("monto_ref")),
            "plazo_ref": _safe_str(r.get("plazo_ref")),
        })
    return estrategias


def generar_diagnostico_gpt(perfil: Dict[str, Any], items: List[Dict[str, Any]], es_garantia: bool) -> str:
    # Si no hay OpenAI, regresa texto base
    if not OPENAI_API_KEY:
        if es_garantia:
            return (
                "Con base en tu perfil y el respaldo patrimonial, identificamos estrategias viables para estructurar "
                "financiamiento bajo CAOM. Un estratega te contactará a la brevedad para validar supuestos y "
                "confirmar la ruta óptima."
            )
        return (
            "Con base en tu perfil, te compartimos recomendaciones preliminares disponibles en nuestra matriz. "
            "En este momento, el alcance es informativo y depende de validación documental posterior."
        )

    contexto = "con garantía inmobiliaria bajo CAOM" if es_garantia else "sin garantía inmobiliaria"

    prompt = f"""
Eres un asesor financiero experto de ASFIES Negocios Consulting.
Redacta un diagnóstico profesional, consultivo y técnico (máximo 2 párrafos).

PERFIL:
- Contacto: {perfil.get("nombre","")} {perfil.get("apellido","")}
- Empresa: {perfil.get("nombre_empresa","")}
- Actividad: {perfil.get("actividad_economica","")}
- Ventas: {perfil.get("ventas_rango","")}
- Tipo de financiamiento: {perfil.get("tipo_financiamiento","")}
- Estado: {contexto}

LISTA (JSON):
{json.dumps(items, ensure_ascii=False)}

INSTRUCCIONES:
- Si hay garantía: habla de “estrategias” con nombre (Alfa/Beta/Delta) y menciona CAOM.
  Cierra con:
  "Atentamente, Miguel Ángel Briseño · consulting@asfiesgroup.com · 55 3573 8572"
  y menciona que contactaremos al correo/teléfono capturados.
- Si NO hay garantía: tono sutil, menciona que por ahora no podemos acompañar el caso como CAOM,
  pero dejamos recomendaciones preliminares. Firma igual: ASFIES Group, Estratega en Financiamiento. Si más datos que ASFIES GROUP en ATENTAMENTE.
""".strip()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        _log(f"OpenAI error: {e}")
        return "Nuestro equipo procesará tu información para entregarte un diagnóstico preliminar."


def send_lead_email(perfil: Dict[str, Any], estrategias: List[Dict[str, Any]]) -> None:
    # Log para confirmar intento
    email_ok = all([SMTP_HOST, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO])
    _log(f"Email attempt? email_configured={email_ok} host={SMTP_HOST} port={SMTP_PORT} from_set={bool(SMTP_FROM)} to_set={bool(SMTP_TO)}")

    if not email_ok:
        _log("Email SKIPPED: faltan variables SMTP_* en Render.")
        return

    to_list = [x.strip() for x in SMTP_TO.split(",") if x.strip()]
    if not to_list:
        _log("Email SKIPPED: SMTP_TO vacío.")
        return

    subject = f"[CAOM Lead] {perfil.get('nombre_empresa','(sin empresa)')} - Garantía inmobiliaria"
    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject

    body = f"""
NUEVO LEAD CAOM (GARANTÍA INMOBILIARIA)

Contacto: {perfil.get('nombre','')} {perfil.get('apellido','')}
Empresa: {perfil.get('nombre_empresa','')}
Teléfono: {perfil.get('telefono','')}
Correo: {perfil.get('correo','')}

Actividad: {perfil.get('actividad_economica','')}
Ventas (rango): {perfil.get('ventas_rango','')}
Tipo financiamiento: {perfil.get('tipo_financiamiento','')}
Monto requerido: {perfil.get('monto_requerido','')}

--- Detalles de inmueble ---
Tipo: {perfil.get('inmueble_tipo','')}
Valor aprox: {perfil.get('inmueble_valor_aprox','')}
Ubicación (Maps): {perfil.get('ubicacion_google_maps','')}

--- Estrategias sugeridas (sin financieras) ---
{json.dumps(estrategias, ensure_ascii=False, indent=2)}
""".strip()

    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=25)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=25)
            server.ehlo()
            server.starttls()
            server.ehlo()

        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, to_list, msg.as_string())
        server.quit()
        _log("Email sent OK.")
    except Exception as e:
        _log(f"Email ERROR: {e}")


@app.get("/health")
def health():
    ok = True
    issues = []

    if not ASFIES_TOKEN:
        ok = False
        issues.append("Falta ASFIES_TOKEN en variables de entorno")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(base_dir, MATRIZ_FILE)):
        ok = False
        issues.append(f"No se encuentra {MATRIZ_FILE} en el repo")

    email_ok = all([SMTP_HOST, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO])
    return jsonify({
        "ok": ok,
        "issues": issues,
        "model": MODEL,
        "matriz_file": MATRIZ_FILE,
        "email_configured": email_ok
    }), 200 if ok else 500


@app.post("/diagnostico")
def diagnostico():
    auth = request.headers.get("Authorization", "")
    if not ASFIES_TOKEN:
        return jsonify({"error": "Server misconfigured: ASFIES_TOKEN missing"}), 500
    if auth != f"Bearer {ASFIES_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    datos = request.get_json(silent=True) or {}

    # Normaliza
    datos["nombre"] = _safe_str(datos.get("nombre"))
    datos["apellido"] = _safe_str(datos.get("apellido"))
    datos["nombre_empresa"] = _safe_str(datos.get("nombre_empresa"))

    es_garantia = _as_bool(datos.get("tiene_garantia_inmueble"))
    solicita_contacto = _as_bool(datos.get("solicita_contacto"))

    _log(f"/diagnostico received | es_garantia={es_garantia} solicita_contacto={solicita_contacto} empresa='{datos.get('nombre_empresa','')}'")

    try:
        financieras = obtener_recomendaciones_financieras(datos)
        _log(f"Recomendaciones encontradas: {len(financieras)}")

        if not financieras:
            header = "Tenemos las siguientes estrategias" if es_garantia else "Opciones de Financiamiento Identificadas"
            return jsonify({
                "status": "not_found",
                "header": header,
                "diagnostico_ia": (
                    "Por el momento no encontramos coincidencias exactas en nuestra matriz para el perfil capturado. "
                    "Si lo deseas, un consultor puede revisarlo manualmente."
                ),
                "recomendaciones": [],
            }), 200

        if es_garantia:
            estrategias = recomendaciones_a_estrategias(financieras)
            diag = generar_diagnostico_gpt(datos, estrategias, es_garantia=True)

            _log("Garantía=TRUE -> intentar envío de correo…")
            # Si quieres mandar SIEMPRE que haya garantía:
            send_lead_email(datos, estrategias)

            return jsonify({
                "status": "success",
                "header": "Tenemos las siguientes estrategias",
                "diagnostico_ia": diag,
                "recomendaciones": estrategias,
            }), 200

        diag = generar_diagnostico_gpt(datos, financieras, es_garantia=False)
        return jsonify({
            "status": "success",
            "header": "Opciones de Financiamiento Identificadas",
            "diagnostico_ia": diag,
            "recomendaciones": financieras,
        }), 200

    except Exception:
        _log("ERROR /diagnostico\n" + traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": "Error interno procesando el diagnóstico",
            "recomendaciones": [],
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
