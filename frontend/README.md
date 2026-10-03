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
npm install
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
   color) y listadas como texto.
3. **Resultado**: anomalía detectada, nivel de confianza (barra
   + %), versión del modelo, «Ver análisis anterior» e
   historial de la sesión (consulta `GET /ecg/{id}`).

El aviso «Herramienta educativa, no es diagnóstico médico» es
siempre visible (encabezado fijo). Los errores del API (422,
404, red) se muestran y anuncian con `aria-live="polite"`.

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
  api/client.js          # cliente de la API (contrato T03)
  components/            # EcgForm, SignalChart, ResultPanel,
                         # HistoryPanel, Disclaimer
  utils/                 # parseSignal, downsample, labels
  test/setup.js
  __tests__/             # pruebas de flujo, formulario,
                         # resultado, a11y (axe) y utilidades
```

## Notas de implementación

- La señal se reduce a ≤1500 puntos para dibujar (promedio de
  bloques); las zonas anómalas (índices de muestra) se dibujan
  1:1 porque el eje X está en índices originales.
- `zonas_anomalas` usa `inicio` inclusivo y `fin` exclusivo.
