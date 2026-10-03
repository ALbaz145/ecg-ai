import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ResultPanel from '../components/ResultPanel.jsx';
import { respuestaAnalisis } from '../test/fixtures.js';

describe('ResultPanel', () => {
  const analisis = respuestaAnalisis({
    anomalia: 'taquicardia',
    confianza: 0.87,
    zonas_anomalas: [
      { inicio: 10, fin: 60, tipo: 'taquicardia' },
    ],
  });

  it('muestra anomalía legible, confianza %, versión y nº de zonas', () => {
    render(<ResultPanel analisis={analisis} />);
    expect(
      screen.getByText('Taquicardia'),
    ).toBeInTheDocument();
    expect(screen.getByText('87 %')).toBeInTheDocument();
    expect(screen.getByText('heuristic-mvp-0.1.0')).toBeInTheDocument();
    expect(
      screen.getByText('Anomalías detectadas'),
    ).toBeInTheDocument();
  });

  it('la barra de confianza es un progressbar accesible', () => {
    render(<ResultPanel analisis={analisis} />);
    const barra = screen.getByRole('progressbar');
    expect(barra).toHaveAttribute('aria-valuenow', '87');
    expect(barra).toHaveAttribute('aria-valuemin', '0');
    expect(barra).toHaveAttribute('aria-valuemax', '100');
    expect(barra).toHaveAttribute(
      'aria-label',
      'Nivel de confianza: 87 por ciento',
    );
  });

  it('el resumen se anuncia con role=status y aria-live', () => {
    render(<ResultPanel analisis={analisis} />);
    const resumen = screen.getByRole('status');
    expect(resumen).toHaveAttribute('aria-live', 'polite');
    expect(resumen).toHaveTextContent('Taquicardia');
    expect(resumen).toHaveTextContent('87');
  });

  it('sin zonas: indica que no hay zonas anómalas', () => {
    render(
      <ResultPanel
        analisis={respuestaAnalisis({
          anomalia: 'normal',
          zonas_anomalas: [],
        })}
      />,
    );
    expect(
      screen.getByText('Normal (sin anomalías detectadas)'),
    ).toBeInTheDocument();
    expect(screen.getByText('Zonas anómalas detectadas')).toBeInTheDocument();
    expect(
      screen.getByText('0'),
    ).toBeInTheDocument();
  });

  it('«Ver análisis anterior» solo aparece si hay anterior y llama al callback', async () => {
    const user = userEvent.setup();
    const onVerAnterior = vi.fn();
    const { rerender } = render(
      <ResultPanel analisis={analisis} />,
    );
    expect(
      screen.queryByRole('button', { name: /Ver análisis anterior/ }),
    ).not.toBeInTheDocument();

    rerender(<ResultPanel analisis={analisis} onVerAnterior={onVerAnterior} />);
    const boton = screen.getByRole('button', {
      name: '← Ver análisis anterior',
    });
    await user.click(boton);
    expect(onVerAnterior).toHaveBeenCalledTimes(1);
  });
});
