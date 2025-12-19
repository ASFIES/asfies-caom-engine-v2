import os
import re
import json
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# =========================
# App
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Config (ENV ONLY)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ASFIES_TOKEN = os.getenv("ASFIES_TOKEN", "").strip()

MODEL = os.getenv("MODEL", "gpt-4o-mini").strip()
MATRIZ_FILE = os.getenv("MATRIZ_FILE", "matriz.xlsx").strip()

# Cache simple de la matriz para no leer el Excel en cada request
_MATRIZ_CACHE: Optional[pd.DataFrame] = None


# =========================
# Utilidades
# =========================
def _log(msg: str):
    print(f"[ASFIES-CAOM] {msg}", flush=True)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _load_matriz() -> pd.DataFrame:
    """
    Carga matriz.xlsx (una sola vez y cachea).
    Debe estar en la raíz del repo junto a app.py.
    """
    global _MATRIZ_CACHE

    if _MATRIZ_CACHE is not None:
        return _MATRIZ_CACHE

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, MATRIZ_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró '{MATRIZ_FILE}' en el repo. Ruta esperada: {path}"
        )

    df = pd.read_excel(path)

    # Normaliza nombres de columnas (por si vienen con espacios raros)
    df.columns = [c.strip() for c in df.columns]

    _MATRIZ_CACHE = df
    _log(f"Matriz cargada: {path} (filas={len(df)}, cols={len(df.columns)})")
    return df


def _extract_min_years(val: Any) -> int:
    """
    Convierte la columna de Antigüedad de la matriz a un mínimo de años (int).
    Si viene algo como "3 años", extrae 3.
    """
    s = _safe_str(val)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _map_antiguedad_user(texto: str) -> int:
    """
    Convierte la antigüedad elegida por el usuario a años (conservador).
    """
    mapa = {
        "Menos de 1 año": 0,
        "Entre 1 y 3 años": 1,
        "Entre 3 y 5 años": 3,
        "Más de 5 años": 5,
    }
    return mapa.get(texto, 0)


def obtener_recomendaciones(perfil: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filtra la matriz según el perfil y retorna top 3 recomendaciones.
    IMPORTANTE: Devuelve lista (puede ser []).
    """
    df = _load_matriz().copy()

    # --- 1) Filtro por Rango de Ventas (si la matriz tiene esa columna exacta)
    rango = _safe_str(perfil.get("ventas_rango"))
    if rango and rango in df.columns:
        df = df[df[rango].astype(str).str.upper().str.strip() == "X"]

    # --- 2) Filtro por Antigüedad
    user_ant = _safe_str(perfil.get("antiguedad"))
    user_years = _map_antiguedad_user(user_ant)

    # Columna puede llamarse 'Antigúedad' (con acento) o 'Antiguedad' (sin acento)
    ant_col = None
    for cand in ["Antigúedad", "Antiguedad", "Antigüedad"]:
        if cand in df.columns:
            ant_col = cand
            break

    if ant_col:
        df["min_anios"] = df[ant_col].apply(_extract_min_years)
        df = df[df["min_anios"] <= user_years]
    else:
        # Si no existe, no filtramos por antigüedad
        df["min_anios"] = 0

    # --- 3) (Opcional) filtro por tipo de financiamiento si la matriz lo soporta
    tipo_user = _safe_str(perfil.get("tipo_financiamiento"))
    # Busca una columna estándar
    tipo_col = None
    for cand in ["Tipo de financiamiento", "Tipo de financiamiento ", "Tipo", "Producto"]:
        if cand in df.columns:
            tipo_col = cand
            break

    if tipo_col and tipo_user:
        # Filtro suave: contiene texto
        df = df[df[tipo_col].astype(str).str.contains(tipo_user, case=False, na=False)]

    # --- Selección top 3
    top_df = df.head(3).copy()
    if top_df.empty:
        return []

    # Mapeo de columnas (si cambian nombres, aquí se ajusta fácil)
    def col(*names: str) -> str:
        for n in names:
            if n in top_df.columns:
                return n
        return ""

    c_fin = col("Financiera", "Nombre", "Institución")
    c_tipo = col("Tipo de financiamiento", "Tipo de financiamiento ", "Tipo", "Producto")
    c_ayu = col("Como ayuda este financiamiento", "Cómo ayuda este financiamiento", "Ayuda", "Descripción")
    c_mto = col("Montos en pesos", "Monto", "Montos")
    c_plz = col("Plazos", "Plazo")

    recs: List[Dict[str, Any]] = []
    for _, row in top_df.iterrows():
        recs.append({
            "financiera": _safe_str(row.get(c_fin)),
            "tipo": _safe_str(row.get(c_tipo)),
            # Tu frontend actual imprime "caracteristicas"
            "caracteristicas": _safe_str(row.get(c_ayu)),
            # Para mostrar también si quieres
            "monto_ref": _safe_str(row.get(c_mto)),
            "plazo_ref": _safe_str(row.get(c_plz)),
        })

    return recs


def generar_diagnostico_gpt(perfil: Dict[str, Any], recomendaciones: List[Dict[str, Any]]) -> str:
    """
    Genera diagnóstico con IA. Si no hay API key o falla, devuelve texto fallback.
    """
    tiene_garantia = bool(perfil.get("tiene_garantia_inmueble"))
    contexto = "con garantía inmobiliaria bajo la metodología CAOM" if tiene_garantia else "sin garantía inmobiliaria"

    # Fallback si no hay key
    if not OPENAI_API_KEY:
        return (
            "Hemos procesado tu información para un diagnóstico preliminar. "
            "Por el momento, las recomendaciones mostradas se basan en la matriz de instituciones aliadas. "
            "Si deseas un análisis consultivo más profundo, un consultor puede revisarlo manualmente."
        )

    prompt = f"""
Eres un asesor financiero experto de ASFIES Negocios Consulting.
Redacta un diagnóstico profesional, consultivo, elegante y técnico (máximo 2 párrafos).

PERFIL:
- Contacto: {perfil.get("nombre","")} {perfil.get("apellido","")}
- Empresa: {perfil.get("nombre_empresa","")}
- Actividad: {perfil.get("actividad_economica","")}
- Ventas: {perfil.get("ventas_rango","")}
- Tipo de financiamiento: {perfil.get("tipo_financiamiento","")}
- Estado: {contexto}

RECOMENDACIONES (lista JSON):
{json.dumps(recomendaciones, ensure_ascii=False)}

INSTRUCCIONES:
- Explica por qué estas opciones son estratégicas para su situación.
- Si hay garantía, menciona que se aplicará la “Arquitectura y Optimización de Capital (CAOM)”.
- Si NO hay garantía, usa un tono sutil para indicar que el alcance es preliminar y puede ser limitado.
""".strip()

    try:
        # OpenAI SDK (openai>=1.x)
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
        return (
            "Nuestro equipo está procesando tu información para un diagnóstico preliminar. "
            "Si deseas un análisis consultivo más profundo, un consultor puede revisarlo manualmente."
        )


# =========================
# Rutas
# =========================
@app.get("/health")
def health():
    ok = True
    issues = []

    if not ASFIES_TOKEN:
        ok = False
        issues.append("Falta ASFIES_TOKEN en variables de entorno")

    # OPENAI_API_KEY puede ser opcional si quieres operar en modo fallback
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), MATRIZ_FILE)):
        ok = False
        issues.append(f"No se encuentra {MATRIZ_FILE} en el repo")

    return jsonify({
        "ok": ok,
        "issues": issues,
        "model": MODEL,
        "matriz_file": MATRIZ_FILE,
    }), 200 if ok else 500


@app.post("/diagnostico")
def diagnostico():
    # --- Seguridad por token
    auth = request.headers.get("Authorization", "")
    if not ASFIES_TOKEN:
        return jsonify({"error": "Server misconfigured: ASFIES_TOKEN missing"}), 500

    if auth != f"Bearer {ASFIES_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    # --- Datos
    datos = request.get_json(silent=True) or {}
    # Normaliza algunos campos esperados
    datos["nombre"] = _safe_str(datos.get("nombre"))
    datos["apellido"] = _safe_str(datos.get("apellido"))
    datos["nombre_empresa"] = _safe_str(datos.get("nombre_empresa"))

    try:
        # 1) Recomendaciones desde matriz
        opciones = obtener_recomendaciones(datos)

        if not opciones:
            # Respuesta consistente para el frontend
            tiene_garantia = bool(datos.get("tiene_garantia_inmueble"))
            header = "Tenemos las siguientes estrategias" if tiene_garantia else "Opciones de Financiamiento Identificadas"

            mensaje = (
                "Por el momento no encontramos coincidencias exactas en nuestra matriz para el perfil capturado. "
                "Podemos revisarlo de manera manual si deseas un análisis más preciso."
            )

            return jsonify({
                "status": "not_found",
                "header": header,
                "diagnostico_ia": mensaje,
                "recomendaciones": [],
            }), 200

        # 2) Diagnóstico IA
        diag_ia = generar_diagnostico_gpt(datos, opciones)

        # 3) Header
        header = "Tenemos las siguientes estrategias" if datos.get("tiene_garantia_inmueble") else "Opciones de Financiamiento Identificadas"

        return jsonify({
            "status": "success",
            "header": header,
            "diagnostico_ia": diag_ia,
            "recomendaciones": opciones or [],
        }), 200

    except Exception as e:
        _log("ERROR /diagnostico\n" + traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": "Error interno procesando el diagnóstico",
            "detail": str(e),
            "recomendaciones": [],
        }), 500


# =========================
# Local run (Render usa gunicorn normalmente)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
