// Configuración global de vitest.
import '@testing-library/jest-dom/vitest';

// jsdom no implementa ResizeObserver (recharts lo usa de forma
// condicional); lo dejamos disponible como no-op por seguridad.
if (typeof window !== 'undefined' && !window.ResizeObserver) {
  window.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
