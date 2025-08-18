#!/usr/bin/env node
/**
 * Beat Addicts Integration Script - CommonJS (.cjs)
 * Supports non-interactive boot (CI/dev) and FAST_DEV lightweight backend init.
 */
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');
const httpProxy = require('http-proxy');
const chalk = require('chalk');
const inquirer = require('inquirer');
const portfinder = require('portfinder');
const dotenv = require('dotenv');
const boxen = require('boxen');
const ora = require('ora');

function parseCliArgs() {
  const args = process.argv.slice(2);
  const flags = { mode: null, noInteractive: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (['--no-interactive','--non-interactive','--ci'].includes(a)) flags.noInteractive = true;
    else if (a === '--mode' && args[i+1]) { flags.mode = args[++i]; }
    else if (a.startsWith('--mode=')) flags.mode = a.split('=')[1];
  }
  if (process.env.NODE_NO_INTERACTIVE === '1') flags.noInteractive = true;
  if (!flags.mode && process.env.MODE) flags.mode = process.env.MODE;
  return flags;
}
const CLI = parseCliArgs();

const DEFAULT_CONFIG = { frontendPort:5173, backendPort:8000, backendHost:'127.0.0.1', proxyPort:3000, environment:'development', exportDir:'exports', generatedAudioDir:'generated_audio', staticDir:'static' };
let frontendProc=null, backendProc=null, proxyServer=null;

function loadConfig(){ if(fs.existsSync('.env')) dotenv.config(); return { frontendPort:+(process.env.FRONTEND_PORT||DEFAULT_CONFIG.frontendPort), backendPort:+(process.env.BACKEND_PORT||DEFAULT_CONFIG.backendPort), backendHost:process.env.BACKEND_HOST||DEFAULT_CONFIG.backendHost, proxyPort:+(process.env.PROXY_PORT||DEFAULT_CONFIG.proxyPort), environment:process.env.NODE_ENV||DEFAULT_CONFIG.environment, exportDir:process.env.EXPORT_DIR||DEFAULT_CONFIG.exportDir, generatedAudioDir:process.env.GENERATED_AUDIO_DIR||DEFAULT_CONFIG.generatedAudioDir, staticDir:process.env.STATIC_DIR||DEFAULT_CONFIG.staticDir }; }
function ensureDirs(c){ const spin=ora('Ensuring directories...').start(); [c.exportDir,c.generatedAudioDir,c.staticDir].forEach(d=>{ if(!fs.existsSync(d)){ fs.mkdirSync(d,{recursive:true}); spin.info('Created '+d); }}); spin.succeed('Directories OK'); }
function writeBackendEnv(c){ const env=`ENVIRONMENT=${c.environment}\nFAST_DEV=1\nEXPORT_DIR=${c.exportDir}\nGENERATED_AUDIO_DIR=${c.generatedAudioDir}\nSTATIC_DIR=${c.staticDir}\nHOST=${c.backendHost}\nPORT=${c.backendPort}\nFRONTEND_URL=http://localhost:${c.proxyPort}\nALLOWED_ORIGINS=http://localhost:${c.proxyPort},http://localhost:${c.frontendPort}`; fs.writeFileSync(path.join('backend','.env'),env); }
function startBackend(c){ const spin=ora('Starting backend...').start(); const activate = fs.existsSync(path.join('backend','venv','bin','activate')) ? 'source venv/bin/activate && ' : ''; backendProc = spawn('bash',['-c',`${activate}python main.py`],{cwd:'backend',stdio:'pipe',shell:true}); let started=false; backendProc.stdout.on('data',d=>{ const out=d.toString(); if(!started && /Uvicorn running on/.test(out)) { started=true; spin.succeed('Backend running'); } process.stdout.write(chalk.green('[Backend] ')+out); }); backendProc.stderr.on('data',d=>process.stderr.write(chalk.red('[Backend ERR] ')+d.toString())); backendProc.on('close',c0=>{ if(c0&&c0!==0) spin.fail('Backend exited code '+c0); }); return backendProc; }
function startFrontend(c){ const spin=ora('Starting frontend...').start(); frontendProc = spawn('npm',['run','dev','--',`--port=${c.frontendPort}`],{stdio:'pipe',shell:true}); let started=false; frontendProc.stdout.on('data',d=>{ const out=d.toString(); if(!started && out.includes('Local:')) { started=true; spin.succeed('Frontend running'); } process.stdout.write(chalk.blue('[Frontend] ')+out); }); frontendProc.stderr.on('data',d=>process.stderr.write(chalk.red('[Frontend ERR] ')+d.toString())); frontendProc.on('close',c0=>{ if(c0&&c0!==0) spin.fail('Frontend exited code '+c0); }); return frontendProc; }
function createProxy(c){ const spin=ora('Creating dev proxy...').start(); const proxy=httpProxy.createProxyServer({ws:true,changeOrigin:true}); proxy.on('error',(e,req,res)=>{ console.error(chalk.red('Proxy error:'),e.message); if(res&&res.writeHead){ res.writeHead(502); res.end('Proxy Error'); }}); const server=http.createServer((req,res)=>{ res.setHeader('Access-Control-Allow-Origin','*'); res.setHeader('Access-Control-Allow-Methods','*'); res.setHeader('Access-Control-Allow-Headers','*'); if(req.method==='OPTIONS'){ res.writeHead(200); res.end(); return; } const backendTargets=['/api/','/docs','/redoc','/openapi.json','/static/','/audio/']; if(backendTargets.some(p=>req.url.startsWith(p))) proxy.web(req,res,{target:`http://${c.backendHost}:${c.backendPort}`}); else proxy.web(req,res,{target:`http://localhost:${c.frontendPort}`}); }); server.on('upgrade',(req,s,h)=>{ if(req.url.startsWith('/ws/')||req.url.startsWith('/api/v1/ws/')) proxy.ws(req,s,h,{target:`ws://${c.backendHost}:${c.backendPort}`}); else proxy.ws(req,s,h,{target:`ws://localhost:${c.frontendPort}`}); }); portfinder.basePort=c.proxyPort; portfinder.getPort((err,port)=>{ if(err){ spin.fail('Proxy port error'); return;} server.listen(port,()=>{ spin.succeed(`Proxy http://localhost:${port}`); console.log(boxen(`${chalk.bold('Beat Addicts Dev Proxy')}\nFrontend: http://localhost:${c.frontendPort}\nBackend:  http://${c.backendHost}:${c.backendPort}\nProxy:    http://localhost:${port}`,{padding:1,borderStyle:'round',borderColor:'green'})); }); proxyServer=server; }); }
function updatePkgScripts(){ const pkgPath='package.json'; if(!fs.existsSync(pkgPath)) return; const pkg=JSON.parse(fs.readFileSync(pkgPath,'utf8')); pkg.scripts = {...pkg.scripts, start:'node connect.cjs', connect:'node connect.cjs', boot:'NODE_NO_INTERACTIVE=1 node connect.cjs --mode start'}; fs.writeFileSync(pkgPath, JSON.stringify(pkg,null,4)); }

async function main(){ console.log(boxen(chalk.bold('Beat Addicts Integration')+'\nCommonJS boot with FAST_DEV',{padding:1,borderStyle:'double',borderColor:'green'})); const cfg=loadConfig(); let mode=CLI.mode; if(CLI.noInteractive && !mode) mode='start'; if(!CLI.noInteractive){ const ans=await inquirer.prompt([{type:'list',name:'mode',message:'Select action',choices:[{name:'Start dev env',value:'start'},{name:'Setup only',value:'setup'},{name:'Exit',value:'exit'}],default:mode||'start'}]); mode=ans.mode; }
  if(mode==='exit') return; if(['setup','start'].includes(mode)){ ensureDirs(cfg); writeBackendEnv(cfg); updatePkgScripts(); console.log(chalk.green('Setup complete.')); }
  if(mode==='start'){ console.log(chalk.cyan('Launching services...')); startBackend(cfg); startFrontend(cfg); createProxy(cfg); if(CLI.noInteractive) console.log(chalk.magenta('Non-interactive mode: Ctrl+C to stop.')); process.on('SIGINT',()=>{ console.log('\nShutting down...'); if(frontendProc) frontendProc.kill(); if(backendProc) backendProc.kill(); if(proxyServer) proxyServer.close(); process.exit(0); }); }
}
main().catch(e=>{ console.error(chalk.red('Fatal:'),e); process.exit(1); });
