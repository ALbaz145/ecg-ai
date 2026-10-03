import { describe, expect, it } from 'vitest';
import { downsampleSignal } from '../utils/downsample.js';

describe('downsampleSignal', () => {
  it('señales pequeñas: sin reducir (paso 1, x = índice)', () => {
    const signal = [0.1, 0.2, 0.3, 0.4];
    const { data, paso, total } = downsampleSignal(signal);
    expect(paso).toBe(1);
    expect(total).toBe(4);
    expect(data).toEqual([
      { x: 0, y: 0.1 },
      { x: 1, y: 0.2 },
      { x: 2, y: 0.3 },
      { x: 3, y: 0.4 },
    ]);
  });

  it('señales grandes: promedia bloques y mantiene índices originales en x', () => {
    const signal = Array.from({ length: 3000 }, (_, i) => i);
    const { data, paso, total } = downsampleSignal(signal, 1500);
    expect(paso).toBe(2);
    expect(total).toBe(3000);
    expect(data.length).toBeLessThanOrEqual(1500);
    // x está en índices de muestra originales → las zonas
    // anómalas (índices) se dibujan 1:1, sin transformación.
    expect(data[0].x).toBe(1); // bloque [0,2) → centro 1
    expect(data[0].y).toBe(0.5); // promedio de 0 y 1
    expect(data[1].x).toBe(3);
    expect(data[data.length - 1].x).toBe(2999);
  });

  it('señal vacía', () => {
    const { data, total } = downsampleSignal([]);
    expect(data).toEqual([]);
    expect(total).toBe(0);
  });
});
