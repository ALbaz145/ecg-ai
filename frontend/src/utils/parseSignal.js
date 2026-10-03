/**
 * Parseo de señales en el cliente (validación previa al envío).
 *
 * - parsePastedValues: texto pegado (valores separados por espacios,
 *   comas, punto y coma o saltos de línea).
 * - parseCsvText: CSV de una columna; replica al backend
 *   (backend/app/pipeline.py::parse_csv_values): primera columna de cada
 *   línea; las filas no numéricas (p. ej. una cabecera) se ignoran.
 */

const SEPARADORES = /[\s,;]+/;

export function parsePastedValues(texto) {
  const valores = [];
  let tokensInvalidos = 0;
  for (const token of String(texto ?? '').split(SEPARADORES)) {
    if (!token) continue;
    const numero = Number(token);
    if (Number.isFinite(numero)) valores.push(numero);
    else tokensInvalidos += 1;
  }
  return { values: valores, invalidTokens: tokensInvalidos };
}

export function parseCsvText(texto) {
  const valores = [];
  for (const linea of String(texto ?? '').split(/\r?\n/)) {
    const primera = linea.trim().split(/[,;\t]/)[0]?.trim();
    if (!primera) continue;
    const numero = Number(primera);
    if (Number.isFinite(numero)) valores.push(numero);
  }
  return { values: valores };
}
