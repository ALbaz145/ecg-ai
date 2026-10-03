import { etiquetaAnomalia } from '../utils/labels.js';

/**
 * Panel de resultado (pantalla 3): anomalía detectada, nivel de
 * confianza (barra + porcentaje), versión del modelo y acceso al
 * análisis anterior.
 */
export default function ResultPanel({ analisis, onVerAnterior }) {
  const pct = Math.round(analisis.confianza * 100);
  const etiqueta = etiquetaAnomalia(analisis.anomalia);

  return (
    <div className="resultado">
      <p className="resultado-resumen" role="status" aria-live="polite">
        Análisis completado: {etiqueta} con una
        confianza del {pct}&nbsp;%.
      </p>

      <dl className="resultado-datos">
        <div className="dato">
          <dt>Anomalías detectadas</dt>
          <dd>{etiqueta}</dd>
        </div>

        <div className="dato">
          <dt>Nivel de confianza</dt>
          <dd>
            <div className="confianza">
              <div
                className="confianza-barra"
                role="progressbar"
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`Nivel de confianza: ${pct} por ciento`}
              >
                <div
                  className="confianza-relleno"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="confianza-texto">{pct} %</span>
            </div>
            <p className="confianza-nota">
              Estimación del modelo heurístico; no está calibrada
              clínicamente.
            </p>
          </dd>
        </div>

        <div className="dato">
          <dt>Versión del modelo</dt>
          <dd>
            <code>{analisis.model_version}</code>
          </dd>
        </div>

        <div className="dato">
          <dt>Zonas anómalas detectadas</dt>
          <dd>{analisis.zonas_anomalas.length}</dd>
        </div>
      </dl>

      {onVerAnterior && (
        <button
          type="button"
          className="boton boton-secundario"
          onClick={onVerAnterior}
        >
          ← Ver análisis anterior
        </button>
      )}
    </div>
  );
}
