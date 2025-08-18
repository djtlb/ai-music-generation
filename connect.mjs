#!/usr/bin/env node
// ESM Dev Orchestrator: backend (FAST_DEV aware) + frontend + proxy
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import http from 'http';
import httpProxy from 'http-proxy';
import chalk from 'chalk';
import inquirer from 'inquirer';
import portfinder from 'portfinder';
import dotenv from 'dotenv';
import boxen from 'boxen';
import ora from 'ora';

function parseArgs(){
  const args=process.argv.slice(2); const cfg={mode:null,ci:false};
  for(let i=0;i<args.length;i++){const a=args[i];
    if(['--no-interactive','--non-interactive','--ci'].includes(a)) cfg.ci=true;
    else if(a==='--mode' && args[i+1]) cfg.mode=args[++i];
    else if(a.startsWith('--mode=')) cfg.mode=a.split('=')[1];
  }
  if(process.env.NODE_NO_INTERACTIVE==='1') cfg.ci=true;
  return cfg;
}
const CLI=parseArgs();

const DEFAULT={frontendPort:5173,backendPort:8000,proxyPort:3000,backendHost:'127.0.0.1'};
function loadConfig(){ if(fs.existsSync('.env')) dotenv.config(); return {
  frontendPort:+(process.env.FRONTEND_PORT||DEFAULT.frontendPort),
  backendPort:+(process.env.BACKEND_PORT||DEFAULT.backendPort),
  proxyPort:+(process.env.PROXY_PORT||DEFAULT.proxyPort),
  backendHost:process.env.BACKEND_HOST||DEFAULT.backendHost,
  fastDev: process.env.FAST_DEV==='0'?0:1
}; }

function ensureEnv(cfg){
  const envPath=path.join('backend','.env');
  if(!fs.existsSync('backend')) return;
  const base=`FAST_DEV=${cfg.fastDev}\nHOST=${cfg.backendHost}\nPORT=${cfg.backendPort}\nALLOWED_ORIGINS=http://localhost:${cfg.proxyPort},http://localhost:${cfg.frontendPort}`;
  fs.writeFileSync(envPath, base, 'utf8');
}

let backendProc=null, frontendProc=null, proxyServer=null;

function startBackend(cfg){
  const spin=ora('Starting backend').start();
  const appTarget = cfg.fastDev ? 'backend.dev_app:app' : 'backend.main:app';
  backendProc = spawn('uvicorn',[appTarget,'--reload','--port',String(cfg.backendPort)],{stdio:'pipe'});
  let up=false;
  backendProc.stdout.on('data',d=>{const t=d.toString(); if(!up && /Uvicorn running/.test(t)){ up=true; spin.succeed('Backend running'); } process.stdout.write(chalk.green('[backend] ')+t);});
  backendProc.stderr.on('data',d=>process.stderr.write(chalk.red('[backend err] ')+d.toString()));
  backendProc.on('exit',c=>{ if(c) console.error(chalk.red('Backend exited code '+c)); });
}

function startFrontend(cfg){
  const spin=ora('Starting frontend').start();
  const args=['run','vite','--','--port',String(cfg.frontendPort)];
  frontendProc=spawn('npm',args,{stdio:'pipe'});
  let up=false;
  frontendProc.stdout.on('data',d=>{const t=d.toString(); if(!up && t.includes('Local:')){ up=true; spin.succeed('Frontend running'); } process.stdout.write(chalk.blue('[frontend] ')+t);});
  frontendProc.stderr.on('data',d=>process.stderr.write(chalk.red('[frontend err] ')+d.toString()));
  frontendProc.on('exit',c=>{ if(c) console.error(chalk.red('Frontend exited code '+c)); });
}

function startProxy(cfg){
  const spin=ora('Starting proxy').start();
  const proxy=httpProxy.createProxyServer({ws:true,changeOrigin:true});
  proxy.on('error',e=>console.error(chalk.red('Proxy error:'),e.message));
  const server=http.createServer((req,res)=>{
    const targetPaths=['/api/','/docs','/redoc','/openapi.json','/static/','/audio/'];
    if(targetPaths.some(p=>req.url.startsWith(p))){
      proxy.web(req,res,{target:`http://${cfg.backendHost}:${cfg.backendPort}`});
    } else {
      proxy.web(req,res,{target:`http://localhost:${cfg.frontendPort}`});
    }
  });
  server.on('upgrade',(req,socket,head)=>{
    if(req.url.startsWith('/ws')) proxy.ws(req,socket,head,{target:`ws://${cfg.backendHost}:${cfg.backendPort}`});
    else proxy.ws(req,socket,head,{target:`ws://localhost:${cfg.frontendPort}`});
  });
  portfinder.basePort=cfg.proxyPort;
  portfinder.getPort((err,port)=>{
    if(err){ spin.fail('Proxy port error'); return; }
    server.listen(port,()=>{
      spin.succeed(`Proxy http://localhost:${port}`);
      console.log(boxen(`${chalk.bold('Dev Environment')}\nBackend:  http://localhost:${cfg.backendPort}\nFrontend: http://localhost:${cfg.frontendPort}\nProxy:    http://localhost:${port}`,{padding:1,borderStyle:'round',borderColor:'green'}));
    });
    proxyServer=server;
  });
}

async function pickMode(initial){
  if(CLI.ci) return initial||'start';
  const ans=await inquirer.prompt([{type:'list',name:'mode',message:'Select action',choices:[{name:'Start dev env',value:'start'},{name:'Exit',value:'exit'}],default:initial||'start'}]);
  return ans.mode;
}

async function main(){
  console.log(boxen(chalk.bold('AI Music Orchestrator (ESM)'),{padding:1,borderStyle:'double',borderColor:'cyan'}));
  const cfg=loadConfig();
  let mode=await pickMode(CLI.mode);
  if(mode==='exit') return;
  ensureEnv(cfg);
  startBackend(cfg);
  startFrontend(cfg);
  startProxy(cfg);
  console.log(chalk.magenta('FAST_DEV='+cfg.fastDev+' (set FAST_DEV=0 for full backend)'));
  if(CLI.ci) console.log(chalk.yellow('Non-interactive mode: Ctrl+C to stop.'));
  process.on('SIGINT',()=>{ console.log('\nShutting down...'); backendProc?.kill(); frontendProc?.kill(); proxyServer?.close(()=>process.exit(0)); setTimeout(()=>process.exit(0),500); });
}
main().catch(e=>{ console.error(chalk.red('Fatal:'),e); process.exit(1); });
