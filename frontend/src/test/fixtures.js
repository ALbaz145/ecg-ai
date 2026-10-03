/**
 * Datos y ayudas compartidos para las pruebas.
 * Sin datos reales de pacientes: señales sintéticas.
 */

/** Señal sintética tipo ECG (onda con picos R artificiales). */
export function senalSintetica(n = 240, { frecuencia = 500 } = {}) {
  const signal = [];
  for (let i = 0; i < n; i += 1) {
    const t = i / frecuencia;
    const latido = Math.sin(2 * Math.PI * 1.2 * t);
    const pico = latido > 0.985 ? 1.1 : 0; // pico R artificial
    signal.push(Number((0.12 * latido + pico).toFixed(4)));
  }
  return signal;
}

/** Respuesta típica de POST /ecg/analyze (contrato T03). */
export function respuestaAnalisis({
  id = '11111111-1111-1111-1111-111111111111',
  ecg_record_id = '22222222-2222-2222-2222-222222222222',
  anomalia = 'taquicardia',
  confianza = 0.87,
  zonas_anomalas = [{ inicio: 10, fin: 60, tipo: 'taquicardia' }],
  model_version = 'heuristic-mvp-0.1.0',
} = {}) {
  return { id, ecg_record_id, anomalia, confianza, zonas_anomalas, model_version };
}

/** Response de fetch con cuerpo JSON. */
export function respuestaJson(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Error de red (fetch lanza). */
export function errorRed() {
  return new TypeError('Failed to fetch');
}

/** CSV de una columna con n muestras (cabecera incluida). */
export function csvSintetico(n = 120) {
  const lineas = ['valor'];
  for (let i = 0; i < n; i += 1) lineas.push((i * 0.01).toFixed(3));
  return lineas.join('\n');
}
