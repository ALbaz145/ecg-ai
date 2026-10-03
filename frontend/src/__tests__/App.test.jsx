import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.jsx';
import {
  respuestaAnalisis,
  respuestaJson,
  senalSintetica,
} from '../test/fixtures.js';

const senal = senalSintetica(240);

/** Llena el formulario y pulsa «Analizar». */
async function analizarDesdeFormulario(user) {
  await user.type(
    screen.getByLabelText('Valores de la señal'),
    senal.join('\n'),
  );
  await user.click(screen.getByRole('button', { name: 'Analizar' }));
}

describe('App — flujo completo (carga → análisis → resultado)', () => {
  let fetchMock;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('analiza y muestra resultado, señal con zonas y aviso permanente', async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async () =>
      respuestaJson(respuestaAnalisis()),
    );
    render(<App />);

    // El aviso obligatorio es visible desde el inicio
    expect(
      screen.getByText('Herramienta educativa, no es diagnóstico médico'),
    ).toBeInTheDocument();

    await analizarDesdeFormulario(user);

    // Petición correcta al contrato de la API
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opciones] = fetchMock.mock.calls[0];
    expect(url).toBe('http://localhost:8000/ecg/analyze');
    expect(opciones.method).toBe('POST');
    expect(JSON.parse(opciones.body)).toEqual({
      signal: senal,
      sample_rate_hz: 500,
      units: 'mV',
    });

    // Resultado (role=status, aria-live polite)
    const resumen = await screen.findByRole('status');
    expect(resumen).toHaveAttribute('aria-live', 'polite');
    expect(resumen).toHaveTextContent('Taquicardia');
    expect(resumen).toHaveTextContent('87');

    // Confianza como barra accesible + porcentaje
    expect(screen.getByRole('progressbar')).toHaveAttribute(
      'aria-valuenow',
      '87',
    );
    expect(screen.getByText('87 %')).toBeInTheDocument();

    // Versión del modelo
    expect(screen.getByText('heuristic-mvp-0.1.0')).toBeInTheDocument();

    // Gráfica accesible (role=img con resumen)
    expect(
      screen.getByRole('img', { name: /240 muestras a 500 Hz/ }),
    ).toBeInTheDocument();

    // Zona anómala marcada también como TEXTO (no solo color)
    expect(screen.getByText('Zonas anómalas (1)')).toBeInTheDocument();
    expect(screen.getByText(/Taquicardia: muestras 10–59/)).toBeInTheDocument();

    // El aviso sigue visible con el resultado
    expect(
      screen.getByText('Herramienta educativa, no es diagnóstico médico'),
    ).toBeInTheDocument();

    // Sin análisis previo: no hay botón «Ver análisis anterior»
    expect(
      screen.queryByRole('button', { name: /Ver análisis anterior/ }),
    ).not.toBeInTheDocument();
  });

  it('muestra y anuncia errores del API 422 (aria-live polite)', async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async () =>
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
    );
    render(<App />);

    await analizarDesdeFormulario(user);

    const region = await screen.findByTestId('region-viva');
    expect(region).toHaveAttribute('aria-live', 'polite');
    expect(region).toHaveTextContent('422');
    expect(region).toHaveTextContent('signal');
    expect(region).toHaveTextContent('Datos no válidos');
  });

  it('muestra y anuncia errores de red', async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(async () => {
      throw new TypeError('Failed to fetch');
    });
    render(<App />);

    await analizarDesdeFormulario(user);

    const region = await screen.findByTestId('region-viva');
    expect(region).toHaveAttribute('aria-live', 'polite');
    expect(region).toHaveTextContent('No se pudo conectar');
  });

  it('historial: dos análisis y «Ver análisis anterior» consulta GET /ecg/{id}', async () => {
    const user = userEvent.setup();
    const primero = respuestaAnalisis({
      id: 'aaaaaaaa-1111-1111-1111-111111111111',
      anomalia: 'taquicardia',
      confianza: 0.8,
    });
    const segundo = respuestaAnalisis({
      id: 'bbbbbbbb-2222-2222-2222-222222222222',
      anomalia: 'normal',
      confianza: 0.65,
      zonas_anomalas: [],
    });
    fetchMock
      .mockResolvedValueOnce(respuestaJson(primero))
      .mockResolvedValueOnce(respuestaJson(segundo))
      .mockImplementation(async () => respuestaJson(primero));
    render(<App />);

    // Primer análisis
    await analizarDesdeFormulario(user);
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Taquicardia',
    );

    // Segundo análisis (el formulario conserva los valores)
    await user.click(screen.getByRole('button', { name: 'Analizar' }));
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Normal (sin anomalías detectadas)',
    );

    // Aparece el botón «Ver análisis anterior»
    const anterior = await screen.findByRole('button', {
      name: '← Ver análisis anterior',
    });
    await user.click(anterior);

    // Consulta GET /ecg/{id} del análisis anterior
    expect(fetchMock).toHaveBeenCalledTimes(3);
    const [urlGet] = fetchMock.mock.calls[2];
    expect(urlGet).toBe(
      'http://localhost:8000/ecg/aaaaaaaa-1111-1111-1111-111111111111',
    );
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Taquicardia',
    );

    // Historial con dos entradas consultables
    expect(
      screen.getByRole('button', {
        name: /Ver análisis 1: Taquicardia, confianza 80 por ciento/,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', {
        name: /Ver análisis 2: Normal/,
      }),
    ).toBeInTheDocument();
  });

  it('historial: pulsar «Ver» de una entrada carga ese análisis (404 anunciado si falla)', async () => {
    const user = userEvent.setup();
    const primero = respuestaAnalisis({
      id: 'aaaaaaaa-1111-1111-1111-111111111111',
      anomalia: 'bradicardia',
      confianza: 0.72,
      zonas_anomalas: [],
    });
    fetchMock
      .mockResolvedValueOnce(respuestaJson(primero))
      .mockImplementation(async () =>
        respuestaJson({ detail: 'Análisis no encontrado.' }, 404),
      );
    render(<App />);

    await analizarDesdeFormulario(user);
    expect(await screen.findByRole('status')).toHaveTextContent(
      'Bradicardia',
    );

    // Pulsar «Ver» en el historial → GET /ecg/{id} → 404
    await user.click(
      screen.getByRole('button', { name: /Ver análisis 1/ }),
    );
    const region = await screen.findByTestId('region-viva');
    expect(region).toHaveTextContent('no encontrado');
  });
});
