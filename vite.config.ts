import react from "@vitejs/plugin-react-swc";
import { defineConfig, PluginOption } from "vite";

import { resolve } from 'node:path'
// Derive project root (Vite provides process.cwd())
const projectRoot = process.env.PROJECT_ROOT || process.cwd();

// https://vite.dev/config/
export default defineConfig({
  // Removed spark + icon proxy plugins (not installed / production simplification)
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(projectRoot, 'src')
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/audio': {
        target: 'http://localhost:8000',
      },
      '/static': {
        target: 'http://localhost:8000',
      }
    },
  },
});
