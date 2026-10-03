import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import EcgForm from '../components/EcgForm.jsx';
import { csvSintetico, senalSintetica } from '../test/fixtures.js';

const senal = senalSintetica(240);

describe('EcgForm', () => {
  let onAnalizar;

  beforeEach(() => {
    onAnalizar = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('todo control tiene label asociado', () => {
    render(<EcgForm onAnalizar={onAnalizar} />);
    // getByLabelText lanza si no hay label asociado
    expect(
      screen.getByLabelText('Valores de la señal'),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText('Frecuencia de muestreo (Hz)'),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('radio', { name: 'Pegar valores' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('radio', { name: 'Subir archivo CSV' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('radio', { name: 'mV (milivoltios)' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('radio', { name: 'µV (microvoltios)' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Analizar' }),
    ).toBeInTheDocument();
  });

  it('navegación completa por teclado (Tab / Shift+Tab)', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);

    await user.tab();
    expect(
      screen.getByRole('radio', { name: 'Pegar valores' }),
    ).toHaveFocus();
    await user.tab();
    expect(
      screen.getByRole('radio', { name: 'Subir archivo CSV' }),
    ).toHaveFocus();
    await user.tab();
    expect(
      screen.getByLabelText('Valores de la señal'),
    ).toHaveFocus();
    await user.tab();
    expect(
      screen.getByLabelText('Frecuencia de muestreo (Hz)'),
    ).toHaveFocus();
    await user.tab();
    expect(
      screen.getByRole('radio', { name: 'mV (milivoltios)' }),
    ).toHaveFocus();
    await user.tab();
    expect(
      screen.getByRole('radio', { name: 'µV (microvoltios)' }),
    ).toHaveFocus();
    await user.tab();
    expect(
      screen.getByRole('button', { name: 'Analizar' }),
    ).toHaveFocus();

    // Shift+Tab vuelve al control anterior
    await user.tab({ shift: true });
    expect(
      screen.getByRole('radio', { name: 'µV (microvoltios)' }),
    ).toHaveFocus();
  });

  it('valida el formulario y anuncia errores (aria-invalid, aria-describedby)', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);

    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    const resumen = await screen.findByRole('alert');
    expect(resumen).toHaveTextContent('Corrige los errores');
    expect(resumen).toHaveTextContent('Pega al menos 100 valores');
    // el foco se mueve al resumen de errores
    expect(resumen).toHaveFocus();

    const valores = screen.getByLabelText('Valores de la señal');
    expect(valores).toHaveAttribute('aria-invalid', 'true');
    const descrito = valores.getAttribute('aria-describedby');
    expect(descrito).toContain('valores-hint');
    expect(descrito).toContain('valores-error');
    expect(
      document.getElementById('valores-error'),
    ).toHaveTextContent('Pega al menos 100 valores');

    expect(onAnalizar).not.toHaveBeenCalled();
  });

  it('valida la frecuencia de muestreo (1–5000 Hz)', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);
    await user.type(
      screen.getByLabelText('Valores de la señal'),
      senal.join('\n'),
    );
    await user.clear(screen.getByLabelText('Frecuencia de muestreo (Hz)'));
    await user.type(
      screen.getByLabelText('Frecuencia de muestreo (Hz)'),
      '9999',
    );
    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    const resumen = await screen.findByRole('alert');
    expect(resumen).toHaveTextContent('entero entre 1 y 5000');
    expect(
      screen.getByLabelText('Frecuencia de muestreo (Hz)'),
    ).toHaveAttribute('aria-invalid', 'true');
    expect(onAnalizar).not.toHaveBeenCalled();
  });

  it('envía la señal pegada con sample_rate_hz y unidades', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);

    await user.type(
      screen.getByLabelText('Valores de la señal'),
      senal.join('\n'),
    );
    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    expect(onAnalizar).toHaveBeenCalledTimes(1);
    const payload = onAnalizar.mock.calls[0][0];
    expect(payload.modo).toBe('texto');
    expect(payload.signal).toHaveLength(240);
    expect(payload.sampleRateHz).toBe(500);
    expect(payload.unidades).toBe('mV');
    expect(payload.filename).toBeNull();
  });

  it('permite cambiar unidades a µV', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);
    await user.type(
      screen.getByLabelText('Valores de la señal'),
      senal.join('\n'),
    );
    await user.click(screen.getByRole('radio', { name: 'µV (microvoltios)' }));
    await user.click(screen.getByRole('button', { name: 'Analizar' }));
    expect(onAnalizar.mock.calls[0][0].unidades).toBe('uV');
  });

  it('modo archivo: sube CSV y envía el archivo con su nombre', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);

    await user.click(
      screen.getByRole('radio', { name: 'Subir archivo CSV' }),
    );
    const archivoInput = screen.getByLabelText(
      'Archivo CSV (una columna)',
    );
    await user.upload(
      archivoInput,
      new File([csvSintetico(120)], 'ecg.csv', { type: 'text/csv' }),
    );
    expect(
      screen.getByText('Archivo seleccionado: ecg.csv'),
    ).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    expect(onAnalizar).toHaveBeenCalledTimes(1);
    const payload = onAnalizar.mock.calls[0][0];
    expect(payload.modo).toBe('archivo');
    expect(payload.archivo.name).toBe('ecg.csv');
    expect(payload.filename).toBe('ecg.csv');
    expect(payload.signal).toHaveLength(120);
    expect(payload.sampleRateHz).toBe(500);
  });

  it('modo archivo: rechaza CSV sin valores suficientes', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);

    await user.click(
      screen.getByRole('radio', { name: 'Subir archivo CSV' }),
    );
    await user.upload(
      screen.getByLabelText('Archivo CSV (una columna)'),
      new File(['valor\n0.1\n0.2\n'], 'corto.csv', {
        type: 'text/csv',
      }),
    );
    await user.click(screen.getByRole('button', { name: 'Analizar' }));

    const resumen = await screen.findByRole('alert');
    expect(resumen).toHaveTextContent('al menos 100');
    expect(onAnalizar).not.toHaveBeenCalled();
  });

  it('muestra el conteo de muestras detectadas al pegar', async () => {
    const user = userEvent.setup();
    render(<EcgForm onAnalizar={onAnalizar} />);
    await user.type(
      screen.getByLabelText('Valores de la señal'),
      senal.join('\n'),
    );
    expect(screen.getByText(/240 muestras detectadas/)).toBeInTheDocument();
  });
});
