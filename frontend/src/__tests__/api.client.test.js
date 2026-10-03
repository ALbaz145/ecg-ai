import { describe, expect, it } from 'vitest';
import {
  analyzeFile,
  analyzeSignal,
  getAnalysis,
  ApiError,
} from '../api/client.js';
import {
  errorRed,
  respuestaAnalisis,
  respuestaJson,
} from '../test/fixtures.js';

function stubFetch(impl) {
  globalThis.fetch = impl;
}

describe('api/client', () => {
  it('analyzeSignal envía JSON con el contrato correcto', async () => {
    const llamadas = [];
    stubFetch(async (url, opciones) => {
      llamadas.push([url, opciones]);
      return respuestaJson(respuestaAnalisis());
    });
    const resultado = await analyzeSignal({
      signal: [0.1, 0.2],
      sampleRateHz: 500,
      unidades: 'mV',
    });
    expect(resultado.anomalia).toBe('taquicardia');
    expect(llamadas[0][0]).toBe(
      'http://localhost:8000/ecg/analyze',
    );
    expect(llamadas[0][1].method).toBe('POST');
    expect(llamadas[0][1].headers['Content-Type']).toBe(
      'application/json',
    );
    expect(JSON.parse(llamadas[0][1].body)).toEqual({
      signal: [0.1, 0.2],
      sample_rate_hz: 500,
      units: 'mV',
    });
  });

  it('analyzeSignal incluye filename solo si existe', async () => {
    const cuerpos = [];
    stubFetch(async (_url, opciones) => {
      cuerpos.push(JSON.parse(opciones.body));
      return respuestaJson(respuestaAnalisis());
    });
    await analyzeSignal({
      signal: [0.1],
      sampleRateHz: 250,
      unidades: 'uV',
      filename: null,
    });
    await analyzeSignal({
      signal: [0.1],
      sampleRateHz: 250,
      unidades: 'uV',
      filename: 'ecg.csv',
    });
    expect(cuerpos[0]).not.toHaveProperty('filename');
    expect(cuerpos[1].filename).toBe('ecg.csv');
  });

  it('analyzeFile envía multipart con file, sample_rate_hz y units', async () => {
    const archivos = new File(['0.1\n0.2\n'], 'ecg.csv', {
      type: 'text/csv',
    });
    let formulario = null;
    stubFetch(async (_url, opciones) => {
      formulario = opciones.body;
      return respuestaJson(respuestaAnalisis());
    });
    await analyzeFile(archivos, 500, 'mV');
    expect(formulario).toBeInstanceOf(FormData);
    expect(formulario.get('file')).toBe(archivos);
    expect(formulario.get('sample_rate_hz')).toBe('500');
    expect(formulario.get('units')).toBe('mV');
  });

  it('getAnalysis usa GET /ecg/{id}', async () => {
    const urls = [];
    stubFetch(async (url) => {
      urls.push(url);
      return respuestaJson(respuestaAnalisis());
    });
    await getAnalysis('abc-123');
    expect(urls[0]).toBe('http://localhost:8000/ecg/abc-123');
  });

  it('mapea errores 422 de Pydantic con el campo y el estado', async () => {
    stubFetch(async () =>
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
    await expect(
      analyzeSignal({ signal: [0.1], sampleRateHz: 500, unidades: 'mV' }),
    ).rejects.toMatchObject({
      name: 'ApiError',
      status: 422,
      message: expect.stringContaining('422'),
    });
    await expect(
      analyzeSignal({ signal: [0.1], sampleRateHz: 500, unidades: 'mV' }),
    ).rejects.toThrow(/signal/);
  });

  it('mapea 404 como análisis no encontrado', async () => {
    stubFetch(async () =>
      respuestaJson({ detail: 'Análisis no encontrado.' }, 404),
    );
    const error = await getAnalysis('no-existe').catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(404);
    expect(error.message).toContain('no encontrado');
  });

  it('mapea 413 (archivo demasiado grande)', async () => {
    stubFetch(async () =>
      respuestaJson({ detail: 'El archivo supera 25000000 bytes.' }, 413),
    );
    const error = await analyzeFile(
      new File(['x'], 'grande.csv'),
      500,
      'mV',
    ).catch((e) => e);
    expect(error.message).toContain('demasiado grande');
  });

  it('anuncia errores de red (servidor no disponible)', async () => {
    stubFetch(async () => {
      throw errorRed();
    });
    const error = await analyzeSignal({
      signal: [0.1],
      sampleRateHz: 500,
      unidades: 'mV',
    }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toContain('No se pudo conectar');
  });
});
