/**
 * Cliente de la API del backend (contrato de T03).
 *
 * - URL base: import.meta.env.VITE_API_URL (por defecto http://localhost:8000).
 * - POST /ecg/analyze        JSON  {signal, sample_rate_hz, units?, filename?}
 * - POST /ecg/analyze/file   multipart (file, sample_rate_hz, units)
 * - GET  /ecg/{id}           análisis almacenado (404 si no existe)
 * - GET  /health             {status: "ok"}
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(status, message, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

/** Formatea el campo `detail` de FastAPI (cadena o errores de Pydantic). */
function detalleTexto(detail) {
  if (Array.isArray(detail)) {
    return detail
      .map((e) => {
        const campo = Array.isArray(e?.loc) ? e.loc.join('.') : e?.loc ?? '';
        return campo ? `${campo}: ${e.msg}` : e?.msg ?? 'error desconocido';
      })
      .join('; ');
  }
  if (typeof detail === 'string' && detail) return detail;
  return 'sin detalles adicionales';
}

function mensajeError(status, detail) {
  switch (status) {
    case 400:
      return `Solicitud incorrecta (400): ${detalleTexto(detail)}`;
    case 404:
      return 'Análisis no encontrado (404). Comprueba el identificador o realiza un análisis nuevo.';
    case 413:
      return 'El archivo es demasiado grande (máximo 25 MB).';
    case 422:
      return `Datos no válidos (422): ${detalleTexto(detail)}`;
    case 500:
      return 'Error interno del servidor (500). Inténtalo de nuevo más tarde.';
    default:
      return `Error inesperado del servidor (${status}): ${detalleTexto(detail)}`;
  }
}

async function request(path, opciones) {
  let respuesta;
  try {
    respuesta = await fetch(`${API_BASE}${path}`, opciones);
  } catch {
    throw new ApiError(
      0,
      `No se pudo conectar con la API en ${API_BASE}. Comprueba que el servidor backend esté en ejecución.`,
    );
  }
  if (!respuesta.ok) {
    let contenido = null;
    try {
      contenido = await respuesta.json();
    } catch {
      /* cuerpo vacío o no JSON */
    }
    const detail = contenido?.detail ?? contenido;
    throw new ApiError(respuesta.status, mensajeError(respuesta.status, detail), detail);
  }
  return respuesta.json();
}

/** POST /ecg/analyze — señal como JSON. */
export async function analyzeSignal({ signal, sampleRateHz, unidades, filename }) {
  const cuerpo = { signal, sample_rate_hz: sampleRateHz, units: unidades };
  if (filename) cuerpo.filename = filename;
  return request('/ecg/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cuerpo),
  });
}

/** POST /ecg/analyze/file — archivo CSV (multipart). */
export async function analyzeFile(archivo, sampleRateHz, unidades) {
  const form = new FormData();
  form.append('file', archivo);
  form.append('sample_rate_hz', String(sampleRateHz));
  form.append('units', unidades);
  return request('/ecg/analyze/file', { method: 'POST', body: form });
}

/** GET /ecg/{id} — recupera un análisis almacenado. */
export async function getAnalysis(id) {
  return request(`/ecg/${encodeURIComponent(id)}`);
}

/** GET /health — comprobación de la API. */
export async function healthCheck() {
  return request('/health');
}
