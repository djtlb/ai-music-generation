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

// State holders
let backendProc=null, frontendProc=null, proxyServer=null;
let backendReady=false, frontendReady=false;
let healthyEmitted=false;

// Helper: bump version
async function bumpVersionPatch(){
  try {
    const pkgRaw=fs.readFileSync('package.json','utf8');
    const pkg=JSON.parse(pkgRaw);
    const [maj,min,pat]=pkg.version.split('.').map(Number);
    const next=`${maj}.${min}.${pat+1}`;
    pkg.version=next;
    fs.writeFileSync('package.json', JSON.stringify(pkg,null,4));
    console.log(chalk.cyan(`Version bumped to ${next} (healthy full boot)`));
  } catch(e){ console.error(chalk.red('Version bump failed:'), e.message); }
}

async function ensureNodeDeps(){
  if(fs.existsSync('node_modules') && fs.existsSync('node_modules/.vite')) return;
  if(!fs.existsSync('package.json')) return;
  const spinner=ora('Installing Node dependencies (npm ci || npm install)').start();
  try {
    const { spawnSync } = await import('child_process');
    let res = spawnSync('npm',['ci','--no-audit','--no-fund'],{stdio:'inherit'});
    if(res.status!==0){ res = spawnSync('npm',['install','--no-audit','--no-fund'],{stdio:'inherit'}); }
    if(res.status===0) spinner.succeed('Node dependencies ready'); else spinner.fail('Node dependency install failed');
  } catch(e){ spinner.fail('Node deps error: '+e.message); }
}

async function ensurePythonEnv(cfg){
  const venvDir='.venv';
  const spinner=ora('Ensuring Python environment').start();
  try {
    if(!fs.existsSync(venvDir)){
      spinner.text='Creating virtual environment';
      const { spawnSync } = await import('child_process');
      const r1=spawnSync('python',['-m','venv',venvDir],{stdio:'inherit'});
      if(r1.status!==0) throw new Error('venv creation failed');
    }
    const py = path.join(venvDir,'bin','python');
    const { spawnSync } = await import('child_process');
    const hasUvicorn=spawnSync(py,['-c','import uvicorn'],{stdio:'ignore'});
    if(hasUvicorn.status!==0){
      spinner.text='Installing core Python deps';
      let req='core-requirements.txt'; if(!fs.existsSync(req)) req='backend/requirements.txt';
      const pipRes=spawnSync(py,['-m','pip','install','-q','-r',req],{stdio:'inherit'});
      if(pipRes.status!==0) throw new Error('pip install failed');
    }
    process.env.PATH=path.join(process.cwd(),venvDir,'bin')+path.delimiter+process.env.PATH;
    spinner.succeed('Python env ready');
  } catch(e){ spinner.fail('Python env issue: '+e.message); }
}

async function pollHealth(cfg){
  const spinner=ora('Waiting for services to become healthy').start();
  const deadline=Date.now()+60_000;
  while(Date.now()<deadline){
    const bOk = await checkBackend(cfg).catch(()=>false);
    const fOk = await checkFrontend(cfg).catch(()=>false);
    backendReady=bOk; frontendReady=fOk;
    spinner.text=`Backend: ${bOk?'✓':'…'}  Frontend: ${fOk?'✓':'…'}`;
    if(bOk && fOk){
      spinner.succeed('All services healthy'); healthyEmitted=true;
      console.log(boxen(`${chalk.green('READY')}\nBackend: http://localhost:${cfg.backendPort}/health\nFrontend: http://localhost:${cfg.frontendPort}\nProxy: (assigned dynamically)`,{padding:1,borderStyle:'classic',borderColor:'green'}));
      if(!cfg.fastDev && CLI.bumpOnGreen){ await bumpVersionPatch(); }
      return;
    }
    await new Promise(r=>setTimeout(r,1000));
  }
  spinner.fail('Services did not become healthy in time');
  if(!backendReady) console.error(chalk.red('Backend not healthy (check logs)'));
  if(!frontendReady) console.error(chalk.red('Frontend not healthy (check logs)'));
}

function checkBackend(cfg){
  return httpGetJson(`http://localhost:${cfg.backendPort}/health`).then(j=> j.status==='ok' || j.status==='healthy');
}
function checkFrontend(cfg){
  return simpleHttp(`http://localhost:${cfg.frontendPort}/`).then(code=> [200,301,302,404].includes(code));
}
function httpGetJson(url){
  return new Promise((res,rej)=>{ try{ http.get(url,r=>{ let data=''; r.on('data',c=>data+=c); r.on('end',()=>{ try{ res(JSON.parse(data)); }catch{ rej(new Error('bad json')); } }); }).on('error',rej); }catch(e){ rej(e);} });
}
function simpleHttp(url){
  return new Promise((res,rej)=>{ try{ http.get(url,r=>{ res(r.statusCode); }).on('error',rej); }catch(e){ rej(e);} });
}

function parseArgs(){
  const args=process.argv.slice(2); const cfg={mode:null,ci:false,full:false,bumpOnGreen:false};
  for(let i=0;i<args.length;i++){const a=args[i];
    if(['--no-interactive','--non-interactive','--ci'].includes(a)) cfg.ci=true;
    else if(a==='--mode' && args[i+1]) cfg.mode=args[++i];
    else if(a.startsWith('--mode=')) cfg.mode=a.split('=')[1];
    else if(a==='--full') cfg.full=true;
    else if(a==='--bump-on-green') cfg.bumpOnGreen=true;
  }
  if(process.env.NODE_NO_INTERACTIVE==='1') cfg.ci=true;
  if(process.env.FAST_DEV==='0') cfg.full=true;
  if(process.env.BUMP_ON_GREEN==='1') cfg.bumpOnGreen=true;
  return cfg;
}
const CLI=parseArgs();

const DEFAULT={frontendPort:5173,backendPort:8000,proxyPort:3000,backendHost:'127.0.0.1'};
function loadConfig(){ if(fs.existsSync('.env')) dotenv.config(); const fastDev = CLI.full ? 0 : (process.env.FAST_DEV==='0'?0:1); return {
  frontendPort:+(process.env.FRONTEND_PORT||DEFAULT.frontendPort),
  backendPort:+(process.env.BACKEND_PORT||DEFAULT.backendPort),
  proxyPort:+(process.env.PROXY_PORT||DEFAULT.proxyPort),
  backendHost:process.env.BACKEND_HOST||DEFAULT.backendHost,
  fastDev
}; }

function ensureEnv(cfg){
  const envPath=path.join('backend','.env');
  if(!fs.existsSync('backend')) return;
  const base=`FAST_DEV=${cfg.fastDev}\nHOST=${cfg.backendHost}\nPORT=${cfg.backendPort}\nALLOWED_ORIGINS=http://localhost:${cfg.proxyPort},http://localhost:${cfg.frontendPort}`;
  fs.writeFileSync(envPath, base, 'utf8');
}

// (state moved to top)

function startBackend(cfg){
  const spin=ora('Starting backend').start();
  const appTarget = cfg.fastDev ? 'backend.dev_app:app' : 'backend.main:app';
  backendProc = spawn('uvicorn',[appTarget,'--reload','--port',String(cfg.backendPort)],{stdio:'pipe'});
  let up=false;
  backendProc.stdout.on('data',d=>{const t=d.toString(); if(!up && /Uvicorn running/.test(t)){ up=true; spin.succeed('Backend process started'); } process.stdout.write(chalk.green('[backend] ')+t);});
  backendProc.stderr.on('data',d=>process.stderr.write(chalk.red('[backend err] ')+d.toString()));
  backendProc.on('exit',c=>{ if(c) console.error(chalk.red('Backend exited code '+c)); });
}

function startFrontend(cfg){
  const spin=ora('Starting frontend').start();
  const args=['run','vite','--','--port',String(cfg.frontendPort)];
  frontendProc=spawn('npm',args,{stdio:'pipe'});
  let up=false;
  frontendProc.stdout.on('data',d=>{const t=d.toString(); if(!up && t.includes('Local:')){ up=true; spin.succeed('Frontend process started'); } process.stdout.write(chalk.blue('[frontend] ')+t);});
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
  await ensurePythonEnv(cfg);
  await ensureNodeDeps();
  startBackend(cfg);
  startFrontend(cfg);
  startProxy(cfg);
  console.log(chalk.magenta(`Mode: ${cfg.fastDev? 'FAST_DEV (lightweight)' : 'FULL'}  (use --full to force full)`));
  // Health polling
  await pollHealth(cfg);
  if(CLI.ci) console.log(chalk.yellow('Non-interactive mode: Ctrl+C to stop.'));
  process.on('SIGINT',()=>{ console.log('\nShutting down...'); backendProc?.kill(); frontendProc?.kill(); proxyServer?.close(()=>process.exit(0)); setTimeout(()=>process.exit(0),500); });
}
main().catch(e=>{ console.error(chalk.red('Fatal:'),e); process.exit(1); });
