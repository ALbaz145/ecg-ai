/**
 * Etiquetas legibles para las claves del modelo (backend/app/model.py
 * y las claves AAMI del agregado en backend/app/inference.py).
 * El modelo devuelve claves en snake_case en español; aquí las hacemos
 * presentables para el personal médico.
 * Las claves desconocidas caen en el fallback `capitalizar`.
 */

export const ANOMALIAS = {
  normal: 'Normal (sin anomalías detectadas)',
  taquicardia: 'Taquicardia',
  bradicardia: 'Bradicardia',
  arritmia: 'Arritmia',
  amplitud_anomala: 'Amplitud anómala del complejo QRS',
  ruido_excesivo: 'Ruido excesivo',
  senal_no_valorable: 'Señal no valorable',
  // Agregado AAMI con modelo real (backend/app/inference.py: AAMI_TO_ANOMALIA → anomalia)
  arritmia_supraventricular: 'Arritmia supraventricular',
  arritmia_ventricular: 'Arritmia ventricular',
  latido_fusion: 'Latido de fusión',
  clase_desconocida: 'Latido de clase desconocida (clase Q, sin clasificar)',
};

export const ZONAS_TIPO = {
  latido_irregular: 'Latido irregular',
  amplitud_baja: 'Amplitud baja',
  amplitud_alta: 'Amplitud alta',
  taquicardia: 'Taquicardia',
  bradicardia: 'Bradicardia',
  ruido_excesivo: 'Ruido excesivo',
  // Una zona por latido no-N (backend/app/inference.py: AAMI_TO_ZONA → tipo)
  latido_supraventricular: 'Latido supraventricular',
  latido_ventricular: 'Latido ventricular',
  latido_fusion: 'Latido de fusión',
  latido_desconocido: 'Latido desconocido',
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
