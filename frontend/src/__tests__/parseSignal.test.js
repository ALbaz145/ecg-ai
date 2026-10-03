import { describe, expect, it } from 'vitest';
import { parseCsvText, parsePastedValues } from '../utils/parseSignal.js';

describe('parseSignal', () => {
  describe('parsePastedValues', () => {
    it('acepta valores separados por salto de línea', () => {
      const { values, invalidTokens } = parsePastedValues(
        '0.1\n0.2\n0.3',
      );
      expect(values).toEqual([0.1, 0.2, 0.3]);
      expect(invalidTokens).toBe(0);
    });

    it('acepta comas, puntos y comas y espacios', () => {
      const { values } = parsePastedValues('0.1, 0.2;0.3 0.4');
      expect(values).toEqual([0.1, 0.2, 0.3, 0.4]);
    });

    it('cuenta (pero ignora) tokens no numéricos', () => {
      const { values, invalidTokens } = parsePastedValues(
        '0.1\nabc\n0.2',
      );
      expect(values).toEqual([0.1, 0.2]);
      expect(invalidTokens).toBe(1);
    });

    it('ignora tokens vacíos', () => {
      const { values } = parsePastedValues('0.1\n\n  \n0.2');
      expect(values).toEqual([0.1, 0.2]);
    });
  });

  describe('parseCsvText (replica al backend)', () => {
    it('ignora la fila de cabecera no numérica', () => {
      const { values } = parseCsvText('valor\n0.1\n0.2\n0.3');
      expect(values).toEqual([0.1, 0.2, 0.3]);
    });

    it('toma solo la primera columna de cada línea', () => {
      const { values } = parseCsvText('0.1,extra\n0.2;extra2');
      expect(values).toEqual([0.1, 0.2]);
    });

    it('ignora líneas vacías y no numéricas', () => {
      const { values } = parseCsvText('\n  \ncabecera\n0.5\n');
      expect(values).toEqual([0.5]);
    });

    it('acepta notación científica y negativos', () => {
      const { values } = parseCsvText('-0.25\n1e-3');
      expect(values).toEqual([-0.25, 0.001]);
    });
  });
});
