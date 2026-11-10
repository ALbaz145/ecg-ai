import { describe, expect, it } from 'vitest';
import {
  ANOMALIAS,
  ZONAS_TIPO,
  etiquetaAnomalia,
  etiquetaTipoZona,
} from '../utils/labels.js';

// Claves AAMI nuevas que emite el backend con modelo real
// (backend/app/inference.py: AAMI_TO_ANOMALIA / AAMI_TO_ZONA → anomalia / tipo).
const ANOMALIAS_AAMI = {
  arritmia_supraventricular: 'Arritmia supraventricular',
  arritmia_ventricular: 'Arritmia ventricular',
  latido_fusion: 'Latido de fusión',
  clase_desconocida: 'Latido de clase desconocida (clase Q, sin clasificar)',
};

const ZONAS_AAMI = {
  latido_supraventricular: 'Latido supraventricular',
  latido_ventricular: 'Latido ventricular',
  latido_fusion: 'Latido de fusión',
  latido_desconocido: 'Latido desconocido',
};

describe('ANOMALIAS (claves nuevas AAMI)', () => {
  it.each(Object.entries(ANOMALIAS_AAMI))(
    '%s → %s',
    (clave, texto) => {
      expect(ANOMALIAS[clave]).toBe(texto);
      expect(etiquetaAnomalia(clave)).toBe(texto);
      // Texto legible: snake_case traducido, sin claves crudas.
      expect(etiquetaAnomalia(clave)).not.toContain('_');
    },
  );

  it('mantiene las etiquetas históricas', () => {
    expect(etiquetaAnomalia('normal')).toBe(
      'Normal (sin anomalías detectadas)',
    );
    expect(etiquetaAnomalia('senal_no_valorable')).toBe(
      'Señal no valorable',
    );
  });
});

describe('ZONAS_TIPO (claves nuevas AAMI)', () => {
  it.each(Object.entries(ZONAS_AAMI))('%s → %s', (clave, texto) => {
    expect(ZONAS_TIPO[clave]).toBe(texto);
    expect(etiquetaTipoZona(clave)).toBe(texto);
    expect(etiquetaTipoZona(clave)).not.toContain('_');
  });

  it('mantiene las etiquetas históricas', () => {
    expect(etiquetaTipoZona('latido_irregular')).toBe('Latido irregular');
    expect(etiquetaTipoZona('ruido_excesivo')).toBe('Ruido excesivo');
  });
});

describe('fallback capitalizar para claves desconocidas', () => {
  it('etiquetaAnomalia capitaliza y sustituye guiones bajos', () => {
    expect(etiquetaAnomalia('clave_inesperada')).toBe('Clave inesperada');
    expect(etiquetaAnomalia('otra')).toBe('Otra');
  });

  it('etiquetaTipoZona capitaliza y sustituye guiones bajos', () => {
    expect(etiquetaTipoZona('zona_misteriosa')).toBe('Zona misteriosa');
    expect(etiquetaTipoZona('singular')).toBe('Singular');
  });

  it('todas las claves de ambos mapas tienen texto sin snake_case', () => {
    for (const clave of Object.keys(ANOMALIAS)) {
      expect(ANOMALIAS[clave]).not.toContain('_');
    }
    for (const clave of Object.keys(ZONAS_TIPO)) {
      expect(ZONAS_TIPO[clave]).not.toContain('_');
    }
  });
});
