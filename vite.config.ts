import react from "@vitejs/plugin-react-swc";
import { defineConfig, PluginOption } from "vite";

import sparkPlugin from "@github/spark/spark-vite-plugin";
import createIconImportProxy from "@github/spark/vitePhosphorIconProxyPlugin";
import { resolve } from 'path'

const projectRoot = process.env.PROJECT_ROOT || import.meta.dirname

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), createIconImportProxy() as PluginOption, sparkPlugin() as PluginOption],
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
