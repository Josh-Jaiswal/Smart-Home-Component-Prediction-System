import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/submit': 'http://localhost:5000',
      '/download_report': 'http://localhost:5000',
      '/static': 'http://localhost:5000'
    }
  },
  build: {
    outDir: 'static/react-build',
    emptyOutDir: true
  }
});