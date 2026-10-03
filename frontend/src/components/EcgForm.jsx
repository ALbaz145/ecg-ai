import { useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { parseCsvText, parsePastedValues } from '../utils/parseSignal.js';

export const MIN_MUESTRAS = 100;
export const MAX_MUESTRAS = 2_000_000;
export const MIN_HZ = 1;
export const MAX_HZ = 5000;

const ESTADO_INICIAL = {
  modo: 'texto', // 'texto' | 'archivo'
  valoresTexto: '',
  archivo: null,
  archivoNombre: '',
  archivoTexto: '',
  sampleRateHz: '500',
  unidades: 'mV',
};

/**
 * Formulario de carga de ECG (pantalla 1).
 *
 * Accesibilidad:
 * - Todo control tiene <label for> asociado.
 * - Ayudas y errores por campo con aria-describedby.
 * - aria-invalid en campos con error.
 * - Al fallar la validación, el resumen de errores recibe el foco.
 */
/**
 * Lee el contenido de texto de un archivo. Usa Blob.text()
 * cuando el entorno lo soporta y FileReader como respaldo
 * (p. ej. jsdom en las pruebas, donde File no tiene .text()).
 */
function leerTextoArchivo(archivo) {
  if (typeof archivo.text === 'function') {
    return archivo.text();
  }
  return new Promise((resolver, rechazar) => {
    const lector = new FileReader();
    lector.onload = () => resolver(String(lector.result));
    lector.onerror = () => rechazar(lector.error);
    lector.readAsText(archivo);
  });
}

export default function EcgForm({ onAnalizar, cargando = false }) {
  const [campos, setCampos] = useState(ESTADO_INICIAL);
  const [errores, setErrores] = useState({});
  const [errorGeneral, setErrorGeneral] = useState('');
  const resumenRef = useRef(null);

  function actualizar(nombre, valor) {
    setCampos((c) => ({ ...c, [nombre]: valor }));
    setErrores((e) => ({ ...e, [nombre]: undefined }));
  }

  async function onArchivoCambio(ev) {
    const archivo = ev.target.files?.[0] ?? null;
    if (!archivo) {
      setCampos((c) => ({
        ...c,
        archivo: null,
        archivoNombre: '',
        archivoTexto: '',
      }));
      return;
    }
    const texto = await leerTextoArchivo(archivo);
    setCampos((c) => ({
      ...c,
      archivo,
      archivoNombre: archivo.name,
      archivoTexto: texto,
    }));
  }

  function validar() {
    const e = {};
    if (campos.modo === 'texto') {
      const { values } = parsePastedValues(campos.valoresTexto);
      if (values.length === 0) {
        e.valores = `Pega al menos ${MIN_MUESTRAS} valores numéricos.`;
      } else if (values.length < MIN_MUESTRAS) {
        e.valores = `Se necesitan al menos ${MIN_MUESTRAS} muestras; has pegado ${values.length}.`;
      } else if (values.length > MAX_MUESTRAS) {
        e.valores = `Máximo ${MAX_MUESTRAS.toLocaleString('es')} muestras; has pegado ${values.length.toLocaleString('es')}.`;
      }
    } else if (!campos.archivo) {
      e.archivo = 'Selecciona un archivo CSV.';
    } else {
      const { values } = parseCsvText(campos.archivoTexto);
      if (values.length === 0) {
        e.archivo = 'El archivo no contiene valores numéricos (¿CSV de una columna?).';
      } else if (values.length < MIN_MUESTRAS) {
        e.archivo = `El archivo tiene ${values.length} muestras; se necesitan al menos ${MIN_MUESTRAS}.`;
      } else if (values.length > MAX_MUESTRAS) {
        e.archivo = `El archivo supera el máximo de ${MAX_MUESTRAS.toLocaleString('es')} muestras.`;
      }
    }
    const hz = Number(campos.sampleRateHz);
    if (!campos.sampleRateHz.trim() || Number.isNaN(hz)) {
      e.frecuencia = 'Introduce la frecuencia de muestreo.';
    } else if (!Number.isInteger(hz) || hz < MIN_HZ || hz > MAX_HZ) {
      e.frecuencia = `Debe ser un número entero entre ${MIN_HZ} y ${MAX_HZ} Hz.`;
    }
    return e;
  }

  function enviar(ev) {
    ev.preventDefault();
    const erroresValidacion = validar();
    if (Object.keys(erroresValidacion).length > 0) {
      // flushSync: el resumen de errores (role="alert") debe
      // existir en el DOM ANTES de moverle el foco.
      flushSync(() => {
        setErrores(erroresValidacion);
        setErrorGeneral('Corrige los errores del formulario antes de analizar.');
      });
      resumenRef.current?.focus();
      return;
    }
    setErrores({});
    setErrorGeneral('');
    const sampleRateHz = Number(campos.sampleRateHz);
    if (campos.modo === 'texto') {
      const { values } = parsePastedValues(campos.valoresTexto);
      onAnalizar({
        modo: 'texto',
        signal: values,
        sampleRateHz,
        unidades: campos.unidades,
        filename: null,
      });
    } else {
      const { values } = parseCsvText(campos.archivoTexto);
      onAnalizar({
        modo: 'archivo',
        archivo: campos.archivo,
        signal: values,
        sampleRateHz,
        unidades: campos.unidades,
        filename: campos.archivoNombre,
      });
    }
  }

  const ids = {
    valoresHint: 'valores-hint',
    valoresError: 'valores-error',
    archivoHint: 'archivo-hint',
    archivoError: 'archivo-error',
    frecuenciaHint: 'frecuencia-hint',
    frecuenciaError: 'frecuencia-error',
  };

  const { values: valoresPrevisual } =
    campos.modo === 'texto'
      ? parsePastedValues(campos.valoresTexto)
      : { values: [] };
  const { invalidTokens } = parsePastedValues(campos.valoresTexto);

  return (
    <form onSubmit={enviar} noValidate aria-labelledby="carga-titulo">
      {errorGeneral && (
        <div
          className="resumen-errores"
          role="alert"
          tabIndex={-1}
          ref={resumenRef}
        >
          <p>
            <strong>{errorGeneral}</strong>
          </p>
          <ul>
            {Object.values(errores)
              .filter(Boolean)
              .map((msg) => (
                <li key={msg}>{msg}</li>
              ))}
          </ul>
        </div>
      )}

      <fieldset className="campo">
        <legend>Método de carga</legend>
        <div className="radios">
          <label className="radio">
            <input
              type="radio"
              name="metodo-texto"
              value="texto"
              checked={campos.modo === 'texto'}
              onChange={() => actualizar('modo', 'texto')}
            />
            Pegar valores
          </label>
          <label className="radio">
            <input
              type="radio"
              name="metodo-archivo"
              value="archivo"
              checked={campos.modo === 'archivo'}
              onChange={() => actualizar('modo', 'archivo')}
            />
            Subir archivo CSV
          </label>
        </div>
      </fieldset>

      {campos.modo === 'texto' ? (
        <div className="campo">
          <label htmlFor="valores">Valores de la señal</label>
          <p id={ids.valoresHint} className="ayuda">
            Un valor por línea o separados por comas. Mínimo {MIN_MUESTRAS}{' '}
            muestras, máximo {MAX_MUESTRAS.toLocaleString('es')}.
          </p>
          <textarea
            id="valores"
            name="valores"
            rows={8}
            value={campos.valoresTexto}
            onChange={(ev) => actualizar('valoresTexto', ev.target.value)}
            aria-required="true"
            aria-invalid={errores.valores ? 'true' : 'false'}
            aria-describedby={`${ids.valoresHint}${errores.valores ? ` ${ids.valoresError}` : ''}`}
          />
          {valoresPrevisual.length > 0 && (
            <p className="ayuda">
              {valoresPrevisual.length.toLocaleString('es')} muestras
              detectadas
              {invalidTokens > 0 &&
                ` · se ignoraron ${invalidTokens} valor(es) no numérico(s)`}
              .
            </p>
          )}
          {errores.valores && (
            <p id={ids.valoresError} className="error-campo">
              {errores.valores}
            </p>
          )}
        </div>
      ) : (
        <div className="campo">
          <label htmlFor="archivo">Archivo CSV (una columna)</label>
          <p id={ids.archivoHint} className="ayuda">
            CSV con una columna de valores numéricos; se acepta una fila de
            cabecera. Tamaño máximo 25 MB.
          </p>
          <input
            type="file"
            id="archivo"
            name="archivo"
            accept=".csv,text/csv"
            onChange={onArchivoCambio}
            aria-describedby={`${ids.archivoHint}${errores.archivo ? ` ${ids.archivoError}` : ''}`}
          />
          {campos.archivoNombre && (
            <p className="ayuda">Archivo seleccionado: {campos.archivoNombre}</p>
          )}
          {errores.archivo && (
            <p id={ids.archivoError} className="error-campo">
              {errores.archivo}
            </p>
          )}
        </div>
      )}

      <div className="campo">
        <label htmlFor="frecuencia">Frecuencia de muestreo (Hz)</label>
        <p id={ids.frecuenciaHint} className="ayuda">
          Número entero entre {MIN_HZ} y {MAX_HZ}. Un ECG típico usa 250–1000
          Hz (por ejemplo, 500).
        </p>
        <input
          type="number"
          id="frecuencia"
          name="frecuencia"
          inputMode="numeric"
          min={MIN_HZ}
          max={MAX_HZ}
          step={1}
          value={campos.sampleRateHz}
          onChange={(ev) => actualizar('sampleRateHz', ev.target.value)}
          aria-required="true"
          aria-invalid={errores.frecuencia ? 'true' : 'false'}
          aria-describedby={`${ids.frecuenciaHint}${errores.frecuencia ? ` ${ids.frecuenciaError}` : ''}`}
        />
        {errores.frecuencia && (
          <p id={ids.frecuenciaError} className="error-campo">
            {errores.frecuencia}
          </p>
        )}
      </div>

      <fieldset className="campo">
        <legend>Unidades de la señal</legend>
        <div className="radios">
          <label className="radio">
            <input
              type="radio"
              name="unidades-mv"
              value="mV"
              checked={campos.unidades === 'mV'}
              onChange={() => actualizar('unidades', 'mV')}
            />
            mV (milivoltios)
          </label>
          <label className="radio">
            <input
              type="radio"
              name="unidades-uv"
              value="uV"
              checked={campos.unidades === 'uV'}
              onChange={() => actualizar('unidades', 'uV')}
            />
            µV (microvoltios)
          </label>
        </div>
      </fieldset>

      <button type="submit" className="boton boton-principal" disabled={cargando}>
        {cargando ? 'Analizando…' : 'Analizar'}
      </button>
    </form>
  );
}
