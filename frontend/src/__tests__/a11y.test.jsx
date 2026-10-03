import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axe from 'axe-core';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import App from '../App.jsx';
import {
  respuestaAnalisis,
  respuestaJson,
  senalSintetica,
} from '../test/fixtures.js';

// Inyecta los estilos reales para que la regla
// color-contrast de axe los evalúe en jsdom.
import estilos from '../styles.css?raw';

const senal = senalSintetica(240);

describe('Accesibilidad (axe, WCAG)', () => {
  beforeAll(() => {
    document.title = 'ECG-AI — Análisis educativo de ECG';
    const estilo = document.createElement('style');
    estilo.textContent = estilos;
    document.head.appendChild(estilo);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('pantalla de carga: sin violaciones de accesibilidad', async () => {
    const { container } = render(<App />);
    const resultados = await axe.run(container);
    expect(resultados.violations).toEqual([]);
  });

  it('pantalla de resultado: sin violaciones de accesibilidad', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        respuestaJson(respuestaAnalisis()),
      ),
    );
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.type(
      screen.getByLabelText('Valores de la señal'),
      senal.join('\n'),
    );
    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    // Espera a que el resultado esté completo
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Taquicardia',
    );

    const resultados = await axe.run(container);
    expect(resultados.violations).toEqual([]);
  });

  it('pantalla de error del API: sin violaciones', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        respuestaJson(
          {
            detail: [
              {
                loc: ['body', 'signal'],
                msg: 'List should have at least 100 items',
              },
            ],
          },
          422,
        ),
      ),
    );
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.type(
      screen.getByLabelText('Valores de la señal'),
      senal.join('\n'),
    );
    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    const region = await screen.findByTestId('region-viva');
    expect(region).toHaveTextContent('422');

    const resultados = await axe.run(container);
    expect(resultados.violations).toEqual([]);
  });
});
