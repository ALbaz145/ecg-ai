import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { etiquetaTipoZona } from '../utils/labels.js';
import { downsampleSignal } from '../utils/downsample.js';

const MAX_PUNTOS = 1500;
const MAX_ETIQUETAS_ZONAS = 20;

function fmtTiempo(indice, sampleRateHz) {
  if (!sampleRateHz) return `${indice}`;
  return `${(indice / sampleRateHz).toFixed(1)}`;
}

/**
 * Gráfico de la señal ECG (pantalla 2).
 *
 * Las zonas anómalas se marcan con sombreado (ReferenceArea)
 * BORDE punteado y ETIQUETA de texto — no solo color. Además,
 * cada zona se lista como texto accesible debajo del gráfico.
 */
export default function SignalChart({
  signal,
  zonasAnomalas = [],
  sampleRateHz,
  unidades,
}) {
  if (!signal || signal.length === 0) {
    return (
      <p>
        La señal de este análisis no está disponible en esta sesión
        (solo se muestra el resultado).
      </p>
    );
  }

  const { data, paso, total } = downsampleSignal(signal, MAX_PUNTOS);
  const resumen = `Gráfico de líneas de la señal ECG: ${total.toLocaleString(
    'es',
  )} muestras a ${sampleRateHz} Hz, en ${unidades}. ${zonasAnomalas.length} zona(s) anómala(s) sombreada(s).`;

  return (
    <figure className="grafico">
      <div
        className="grafico-contenedor"
        role="img"
        aria-label={resumen}
      >
        <LineChart
          width={900}
          height={320}
          data={data}
          margin={{ top: 28, right: 24, bottom: 40, left: 12 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#d1d5db"
            vertical={false}
          />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, total - 1]}
            tickFormatter={(v) => fmtTiempo(v, sampleRateHz)}
            label={{
              value: 'Tiempo (s)',
              position: 'insideBottom',
              offset: -24,
            }}
          />
          <YAxis
            width={72}
            label={{
              value: `Amplitud (${unidades})`,
              angle: -90,
              position: 'insideLeft',
            }}
          />
          <Tooltip
            labelFormatter={(v) =>
              `Muestra ${Math.round(v)} (${fmtTiempo(v, sampleRateHz)} s)`
            }
          />
          {zonasAnomalas.map((zona, i) => (
            <ReferenceArea
              key={`${zona.inicio}-${zona.fin}-${i}`}
              x1={zona.inicio}
              x2={Math.min(zona.fin, total)}
              fill="#fef3c7"
              fillOpacity={0.6}
              stroke="#b45309"
              strokeDasharray="5 4"
              label={
                zonasAnomalas.length <= MAX_ETIQUETAS_ZONAS
                  ? {
                      value: `⚠ ${etiquetaTipoZona(zona.tipo)}`,
                      position: 'top',
                      fill: '#78350f',
                      fontSize: 12,
                    }
                  : undefined
              }
            />
          ))}
          <Line
            type="monotone"
            dataKey="y"
            stroke="#1d4ed8"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </div>
      <figcaption className="grafico-nota">
        {paso > 1
          ? `Visualización reducida: promedio de bloques de ${paso} muestras (${data.length.toLocaleString(
              'es',
            )} de ${total.toLocaleString('es')} puntos). `
          : `${total.toLocaleString('es')} muestras. `}
        Las zonas anómalas aparecen sombreadas, con borde punteado y
        etiqueta de texto (no solo por color), y se listan a
        continuación.
      </figcaption>
      <div className="zonas" aria-labelledby="zonas-titulo">
        <h3 id="zonas-titulo">
          Zonas anómalas ({zonasAnomalas.length})
        </h3>
        {zonasAnomalas.length === 0 ? (
          <p className="zonas-vacia">
            No se detectaron zonas anómalas en esta señal.
          </p>
        ) : (
          <ol className="zonas-lista">
            {zonasAnomalas.map((zona, i) => (
              <li key={`${zona.inicio}-${zona.fin}-${i}`}>
                <span aria-hidden="true">⚠ </span>
                <strong>
                  {etiquetaTipoZona(zona.tipo)}
                  {': '}muestras {zona.inicio}–
                  {Math.max(zona.inicio, zona.fin - 1)}
                </strong>
                {' '}
                ({(zona.fin - zona.inicio).toLocaleString('es')} muestras;
                índice fin {zona.fin} exclusivo)
              </li>
            ))}
          </ol>
        )}
      </div>
    </figure>
  );
}
