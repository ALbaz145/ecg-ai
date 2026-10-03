import { describe, expect, it } from 'vitest';

/**
 * Verificación de los ratios de contraste de la paleta
 * (WCAG 2.1 AA):
 *  - Texto normal: ≥ 4.5:1
 *  - Componentes UI / indicadores de foco / gráficos
 *    no textuales: ≥ 3:1
 */

function luminancia(hex) {
  const c = hex.replace('#', '');
  const rgb = [0, 2, 4]
    .map((i) => parseInt(c.substring(i, i + 2), 16) / 255)
    .map((v) =>
      v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4,
    );
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function contraste(hexA, hexB) {
  const l1 = luminancia(hexA);
  const l2 = luminancia(hexB);
  const claro = Math.max(l1, l2);
  const oscuro = Math.min(l1, l2);
  return (claro + 0.05) / (oscuro + 0.05);
}

describe('Contraste WCAG AA de la paleta', () => {
  const casosTexto = [
    ['#111827', '#ffffff', 'texto del body'],
    ['#374151', '#ffffff', 'texto secundario'],
    ['#ffffff', '#1d4ed8', 'botón principal'],
    ['#ffffff', '#6b7280', 'botón deshabilitado'],
    ['#b91c1c', '#ffffff', 'error sobre blanco'],
    ['#b91c1c', '#fef2f2', 'error sobre su fondo'],
    ['#166534', '#ffffff', 'éxito sobre blanco'],
    ['#166534', '#f0fdf4', 'éxito sobre su fondo'],
    ['#78350f', '#fef3c7', 'aviso (disclaimer)'],
    ['#78350f', '#ffffff', 'etiqueta de zona en gráfico'],
    ['#1d4ed8', '#ffffff', 'enlaces'],
    ['#1d4ed8', '#eff6ff', 'enlace sobre fondo claro'],
    ['#111827', '#f0fdf4', 'resumen del resultado'],
  ];

  it.each(casosTexto)(
    'texto %s sobre %s (%s) ≥ 4.5:1',
    (frente, fondo, nombre) => {
      expect(
        contraste(frente, fondo),
        `${nombre}: ${frente} sobre ${fondo}`,
      ).toBeGreaterThanOrEqual(4.5);
    },
  );

  const casosUI = [
    ['#4b5563', '#ffffff', 'bordes de controles'],
    ['#000000', '#ffffff', 'foco sobre blanco'],
    ['#000000', '#1d4ed8', 'foco sobre botón azul'],
    ['#b45309', '#fef3c7', 'borde de zona anómala'],
    ['#1d4ed8', '#ffffff', 'línea de la gráfica'],
    ['#b45309', '#ffffff', 'zona anómala sobre blanco'],
  ];

  it.each(casosUI)(
    'componente %s sobre %s (%s) ≥ 3:1',
    (frente, fondo, nombre) => {
      expect(
        contraste(frente, fondo),
        `${nombre}: ${frente} sobre ${fondo}`,
      ).toBeGreaterThanOrEqual(3);
    },
  );
});
