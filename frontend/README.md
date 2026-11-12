# ECG-AI — Frontend (React + Vite)

Interfaz accesible (WCAG AA) para personal médico: carga un ECG,
lo analiza y muestra el resultado.

> ⚠️ **Herramienta educativa, no es diagnóstico médico.**

## Requisitos

- Node 20+ y npm.
- API del backend en `VITE_API_URL` (por defecto
  `http://localhost:8000`; ver `.env.example`).

## Uso

```bash
npm install        # (o `npm ci` para una instalación limpia y fija)
npm run dev        # desarrollo en http://localhost:5173
npm run build      # build de producción → dist/
npm run preview    # sirve dist/
npm test           # pruebas (vitest + Testing Library + axe)
npm run lint       # ESLint (incluye reglas jsx-a11y)
```

## Pantallas

1. **Carga de ECG**: pegar valores o subir CSV (una columna),
   frecuencia de muestreo (Hz) y unidades (mV/µV).
2. **Señal**: gráfica de la señal con las **zonas anómalas
   sombreadas** (borde punteado + etiqueta de texto, no solo
   color; con más de 20 zonas la etiqueta se omite dentro de la
   gráfica) y listadas siempre como texto debajo del gráfico.
3. **Resultado**: anomalía detectada, nivel de confianza (barra
   + %), versión del modelo (`model_version`, p. ej.
   `heuristic-mvp-0.1.0` o `baseline-sklearn-0.2.0`) y
   «Ver análisis anterior».
4. **Historial** (de la sesión): botones que vuelven a consultar
   la API (`GET /ecg/{id}`) y reabren ese resultado.

Las etiquetas de anomalía y de zona mostradas pueden venir del
**modelo heurístico** o del **modelo real (clases AAMI)**, según
el artefacto que cargue el backend; en ambos casos las traduce
`utils/labels.js` a texto legible y las claves desconocidas caen
en el fallback `capitalizar` (una clave nueva no rompe la UI).

El aviso «Herramienta educativa, no es diagnóstico médico» es
siempre visible (encabezado fijo + pie; componente
`Disclaimer`). Los errores del API (400, 404, 413, 422, 500 y
de red) se muestran y anuncian con `aria-live="polite"`.

## Accesibilidad (WCAG AA)

- Labels asociados a todos los controles; errores con
  `aria-describedby` + `aria-invalid`; resumen de errores
  enfocable.
- Navegación completa por teclado (skip link, focus visible).
- Contraste AA verificado: pruebas automáticas con axe
  (incluido `color-contrast`) y prueba unitaria de ratios.
- Gráfica con `role="img"` + resumen textual y lista de zonas
  (la información no depende solo del color).

## Estructura

```
src/
  App.jsx              # ensambla las 4 pantallas y los avisos
  main.jsx             # punto de entrada React
  styles.css           # estilos globales (tema, foco, gráfica)
  api/client.js        # cliente de la API (contrato T03)
  components/          # EcgForm, SignalChart, ResultPanel,
                       # HistoryPanel, Disclaimer
  utils/               # parseSignal, downsample, labels
  test/                # setup.js (vitest) y fixtures.js
  __tests__/           # flujo (App), formulario, resultado,
                       # a11y (axe), contraste, cliente API,
                       # parseSignal, downsample y labels
```

## Notas de implementación

- La señal se reduce a ≤1500 puntos para dibujar (promedio de
  bloques); las zonas anómalas (índices de muestra) se dibujan
  1:1 porque el eje X está en índices originales (formateados
  en segundos).
- `zonas_anomalas` usa `inicio` inclusivo y `fin` exclusivo.
- `utils/labels.js` mapea **ambos vocabularios** que puede
  emitir el backend: el heurístico (`taquicardia`,
  `ruido_excesivo`, …) y el agregado AAMI del modelo real de
  `backend/app/inference.py` (`arritmia_supraventricular`,
  `latido_ventricular`, …). `etiquetaAnomalia` y
  `etiquetaTipoZona` traducen la clave y aplican el fallback
  `capitalizar` cuando no la conocen.
- El aviso educativo vive en `components/Disclaimer.jsx` (y se
  repite en el pie de `App.jsx`).
