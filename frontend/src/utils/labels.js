/**
 * Etiquetas legibles para las claves del modelo (backend/app/model.py).
 * El modelo devuelve claves en snake_case en español; aquí las hacemos
 * presentables para el personal médico.
 */

export const ANOMALIAS = {
  normal: 'Normal (sin anomalías detectadas)',
  taquicardia: 'Taquicardia',
  bradicardia: 'Bradicardia',
  arritmia: 'Arritmia',
  amplitud_anomala: 'Amplitud anómala del complejo QRS',
  ruido_excesivo: 'Ruido excesivo',
  senal_no_valorable: 'Señal no valorable',
};

export const ZONAS_TIPO = {
  latido_irregular: 'Latido irregular',
  amplitud_baja: 'Amplitud baja',
  amplitud_alta: 'Amplitud alta',
  taquicardia: 'Taquicardia',
  bradicardia: 'Bradicardia',
  ruido_excesivo: 'Ruido excesivo',
};

function capitalizar(texto) {
  return texto.charAt(0).toUpperCase() + texto.slice(1);
}

export function etiquetaAnomalia(clave) {
  return ANOMALIAS[clave] ?? capitalizar(String(clave).replace(/_/g, ' '));
}

export function etiquetaTipoZona(clave) {
  return ZONAS_TIPO[clave] ?? capitalizar(String(clave).replace(/_/g, ' '));
}
