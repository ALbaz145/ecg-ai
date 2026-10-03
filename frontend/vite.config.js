/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  test: {
    environment: 'jsdom',
    globals: true, // permite la auto-limpieza de @testing-library/react
    setupFiles: ['./src/test/setup.js'],
    // Las pruebas de UI (jsdom + axe + escritura tecla a tecla
    // con user-event) son intensivas en CPU. Ejecutar los archivos
    // en paralelo sobrecarga el bucle de eventos y provoca
    // timeouts y carreras; en secuencial son deterministas.
    fileParallelism: false,
    // Escribir ~1400 teclas en jsdom tarda >4 s por test;
    // el límite por defecto (5 s) es demasiado ajustado.
    testTimeout: 20000,
  },
});
