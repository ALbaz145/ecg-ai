import { etiquetaAnomalia } from '../utils/labels.js';

/**
 * Historial de la sesión (pantalla 3, parte): lista de análisis
 * realizados. Cada botón consulta la API (GET /ecg/{id}) y muestra
 * ese análisis.
 */
export default function HistoryPanel({ historial, activoId, onVer }) {
  return (
    <div className="historial">
      <p id="historial-desc" className="ayuda">
        Análisis realizados en esta sesión. Al pulsar «Ver» se consulta
        la API (GET /ecg/…) y se muestra ese resultado.
      </p>
      <ul className="historial-lista" aria-describedby="historial-desc">
        {historial.map((h, i) => {
          const activo = h.id === activoId;
          return (
            <li key={h.id}>
              <button
                type="button"
                onClick={() => onVer(h.id)}
                aria-current={activo ? 'true' : undefined}
                aria-label={`Ver análisis ${i + 1}: ${etiquetaAnomalia(
                  h.anomalia,
                )}, confianza ${Math.round(h.confianza * 100)} por ciento${
                  h.filename ? `, archivo ${h.filename}` : ''
                }`}
              >
                <span className="historial-num" aria-hidden="true">
                  #{i + 1}
                </span>
                <span className="historial-info">
                  <strong>{etiquetaAnomalia(h.anomalia)}</strong>
                  <span>
                    {Math.round(h.confianza * 100)} % ·{' '}
                    {new Date(h.fechaISO).toLocaleString('es')}
                    {h.filename ? ` · ${h.filename}` : ''}
                    {activo ? ' · (actual)' : ''}
                  </span>
                </span>
                <span className="historial-accion">Ver</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
