import { useCallback, useState } from 'react';
import { analyzeFile, analyzeSignal, getAnalysis } from './api/client.js';
import Disclaimer from './components/Disclaimer.jsx';
import EcgForm from './components/EcgForm.jsx';
import SignalChart from './components/SignalChart.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import HistoryPanel from './components/HistoryPanel.jsx';
import { etiquetaAnomalia } from './utils/labels.js';

/**
 * Aplicación principal: carga → análisis → resultado.
 *
 * El aviso «Herramienta educativa, no es diagnóstico médico»
 * (Disclaimer) vive en el encabezado fijo y está SIEMPRE
 * visible. Los errores y mensajes de estado se anuncian en
 * una región aria-live="polite".
 */
export default function App() {
  const [resultado, setResultado] = useState(null);
  const [historial, setHistorial] = useState([]);
  const [cargando, setCargando] = useState(false);
  const [mensaje, setMensaje] = useState('');
  const [error, setError] = useState('');

  const handleAnalizar = useCallback(async (datos) => {
    setCargando(true);
    setError('');
    setMensaje('Analizando la señal…');
    try {
      const respuesta =
        datos.modo === 'archivo'
          ? await analyzeFile(
              datos.archivo,
              datos.sampleRateHz,
              datos.unidades,
            )
          : await analyzeSignal({
              signal: datos.signal,
              sampleRateHz: datos.sampleRateHz,
              unidades: datos.unidades,
              filename: datos.filename,
            });
      const entrada = {
        ...respuesta,
        signal: datos.signal,
        sampleRateHz: datos.sampleRateHz,
        unidades: datos.unidades,
        filename: datos.filename || null,
        fechaISO: new Date().toISOString(),
      };
      setHistorial((prev) => [...prev, entrada]);
      setResultado(entrada);
      setMensaje(
        `Análisis completado: ${etiquetaAnomalia(
          respuesta.anomalia,
        )} con ${Math.round(respuesta.confianza * 100)} % de confianza.`,
      );
    } catch (err) {
      setError(err.message);
      setMensaje('');
    } finally {
      setCargando(false);
    }
  }, []);

  const handleVerAnalisis = useCallback(
    async (id) => {
      setCargando(true);
      setError('');
      setMensaje('Cargando el análisis solicitado…');
      try {
        const respuesta = await getAnalysis(id);
        const local = historial.find((h) => h.id === id);
        setResultado({
          ...respuesta,
          signal: local ? local.signal : null,
          sampleRateHz: local ? local.sampleRateHz : null,
          unidades: local ? local.unidades : 'mV',
          filename: local ? local.filename : null,
          fechaISO: local ? local.fechaISO : new Date().toISOString(),
        });
        setMensaje(
          `Análisis cargado: ${etiquetaAnomalia(
            respuesta.anomalia,
          )} con ${Math.round(respuesta.confianza * 100)} % de confianza.`,
        );
      } catch (err) {
        setError(err.message);
        setMensaje('');
      } finally {
        setCargando(false);
      }
    },
    [historial],
  );

  const indiceActual = resultado
    ? historial.findIndex((h) => h.id === resultado.id)
    : -1;
  const hayAnterior = indiceActual > 0;
  const verAnterior = hayAnterior
    ? () => handleVerAnalisis(historial[indiceActual - 1].id)
    : undefined;

  return (
    <div className="app">
      <a className="saltar" href="#contenido-principal">
        Saltar al contenido principal
      </a>

      <header className="encabezado">
        <div className="encabezado-inner contenedor">
          <div>
            <h1>ECG-AI</h1>
            <p className="tagline">
              Análisis educativo de electrocardiogramas
            </p>
          </div>
        </div>
        <Disclaimer />
      </header>

      <main
        id="contenido-principal"
        className="contenedor contenido"
        aria-busy={cargando}
      >
        {/* Región aria-live: errores del API (422, 404, red)
            y mensajes de estado se anuncian aquí. */}
        <div
          className="region-viva"
          aria-live="polite"
          data-testid="region-viva"
        >
          {error ? (
            <p className="aviso-error">{error}</p>
          ) : mensaje ? (
            <p className="aviso-info">{mensaje}</p>
          ) : null}
        </div>

        <section className="seccion" aria-labelledby="carga-titulo">
          <h2 id="carga-titulo">1. Carga de ECG</h2>
          <EcgForm onAnalizar={handleAnalizar} cargando={cargando} />
        </section>

        {resultado && (
          <>
            <section
              className="seccion"
              aria-labelledby="senal-titulo"
            >
              <h2 id="senal-titulo">2. Señal ECG</h2>
              <SignalChart
                signal={resultado.signal}
                zonasAnomalas={resultado.zonas_anomalas}
                sampleRateHz={resultado.sampleRateHz}
                unidades={resultado.unidades}
              />
            </section>

            <section
              className="seccion"
              aria-labelledby="resultado-titulo"
            >
              <h2 id="resultado-titulo">3. Resultado</h2>
              <ResultPanel
                analisis={resultado}
                onVerAnterior={verAnterior}
              />
            </section>
          </>
        )}

        {historial.length > 0 && (
          <section
            className="seccion"
            aria-labelledby="historial-titulo"
          >
            <h2 id="historial-titulo">
              4. Historial de análisis (esta sesión)
            </h2>
            <HistoryPanel
              historial={historial}
              activoId={resultado?.id}
              onVer={handleVerAnalisis}
            />
          </section>
        )}
      </main>

      <footer className="pie contenedor">
        <p>
          <strong>Herramienta educativa, no es diagnóstico
          médico.</strong> Los resultados del modelo son
          heurísticos, no están calibrados y no sustituyen el
          criterio de un profesional sanitario.
        </p>
        <p>
          ECG-AI · Proyecto escolar · Datos de prueba públicos
          (PTB-XL / MIT-BIH), sin datos reales de pacientes.
        </p>
      </footer>
    </div>
  );
}
