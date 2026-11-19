const $ = id => document.getElementById(id);
const descriptions = {
  clean:'Exact outlines and unaltered colors. The starting point for every experiment.',
  shrink_1px_half:'Half the piece outlines shrink by one pixel. A small change with a revealing effect.',
  uneven_outline:'Irregular outline errors test matching beyond a perfect geometric fit.',
  smoothed_outline:'Softened outlines round off corners and tabs while preserving the picture.',
  pixel_noise:'Noise added to the colors tests how well the model handles imperfect imagery.',
  flat_even:'Straight seams, no tabs. A test of how local matches become a whole picture.',
  flat_diagonal:'Slanted seams and irregular quadrilaterals. The same image, a different challenge.',
  color_removed:'Color is removed entirely. A diagnostic for how much the model relies on shape.'
};
const icons = ['▦','▧','〰','◌','⁙','⊞','⋈','◐'];
let meta, category='flat_even', anchored=true, data=null, atlas=null, frame=0, timer=null, request=0;
let mode='validation',customSource='gallery-0',customName='Gallery image 1',uploading=false;
let failures=[],failureOpen=false,failureRequest=0;
let strategy='grid';
let wrongSlots=[];
let updaterData=null,updaterRequest=0;
const delay = ms => new Promise(resolve => setTimeout(resolve,ms));
const selection = () => ({model:$('model').value,category,index:mode==='custom'?0:Number($('image-number').value)-1,anchor:anchored&&category.startsWith('flat_')?1:0,strategy:category.startsWith('flat_')?strategy:'grid',...(mode==='custom'?{source:customSource,grid:$('grid-size').value}:{})});
const isFree = s => s==='strongest'||s==='shrink';
const pct = x => `${(x*100).toFixed(1)}%`;
function stop(){clearTimeout(timer);timer=null;$('play').textContent='▶';$('play').setAttribute('aria-label','Play assembly');}
function anchorUI(){
  const flat=category.startsWith('flat_');
  $('anchor-control').hidden=!flat;
  $('anchor-note').textContent=flat?(strategy==='shrink'?'Strongest-pair growth, then a frame around it shrinks one side at a time to the board size; evicted pieces move to their best empty cell inside. A swap polish follows. Uses the piece count for the board size, never the layout.':strategy==='strongest'?'Strongest pair seeds an open board. Growth uses matching scores only; no corner or rectangular boundary is supplied.':anchored?'Anchored evaluation supplies the true top-left piece. All other placements are inferred.':'Unanchored evaluation. The solver chooses its own starting piece.'):mode==='custom'?'Regular jigsaw cuts with known board dimensions and upright pieces.':'Same held-out images. Outlines and colors change according to the selected category.';
}
function setCategory(id){
  category=id;
  document.querySelectorAll('.category').forEach(b=>{b.classList.toggle('active',b.dataset.id===id);b.setAttribute('aria-current',b.dataset.id===id?'page':'false');});
  $('category-title').textContent=mode==='custom'?'Try an image':meta.categories.find(c=>c.id===id).label;
  $('category-description').textContent=mode==='custom'?'Choose a picture, a cut, and a model. See what comes together.':descriptions[id];anchorUI();load();
  if(mode==='custom')$('cut-type').value=id;
}
function controls(enabled){for(const id of ['play','restart','step-back','step-forward','scrubber','speed'])$(id).disabled=!enabled;}
function setLoading(message){$('loading').hidden=false;$('loading-title').textContent='Preparing your puzzle';$('loading-message').textContent=message;$('retry').hidden=true;document.querySelector('.spinner').hidden=false;document.querySelector('.viewer').setAttribute('aria-busy','true');}
async function fetchJSON(url){const r=await fetch(url);const body=await r.json();if(!r.ok)throw new Error(body.message||`Request failed (${r.status})`);return body;}
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Could not load puzzle artwork. Please retry.'));image.src=url;});}
async function load(){
  if(uploading)return;
  const s=selection();
  if(!Number.isInteger(s.index)||s.index<0||s.index>=100){$('image-number').reportValidity();return;}
  const ticket=++request;stop();controls(false);data=null;
  updaterData=null;++updaterRequest;$('updater-analysis').hidden=true;
  wrongSlots=[];$('incorrect-legend').hidden=true;
  const free=isFree(s.strategy);
  $('free-metrics').hidden=!free;
  $('proposed-legend').hidden=free;
  for(const id of ['free-placed','free-recall','free-bounds'])$(id).textContent='—';
  $('location-label').textContent=free?'ALIGNED ACCURACY':'PLACEMENT ACCURACY';
  $('neighbor-label').textContent=free?'NEIGHBOR PRECISION':'NEIGHBORS';
  document.querySelectorAll('.timeline-labels span').forEach((el,i)=>el.textContent=(s.strategy==='shrink'?['STRONGEST PAIR','SHRINKING FRAME','SWAP POLISH']:free?['STRONGEST PAIR','OUTWARD GROWTH','FREE ASSEMBLY']:['INITIAL PLACEMENT','REFINEMENT','FINAL ASSIGNMENT'])[i]);
  refreshFailures();
  $('reference').hidden=true;for(const id of ['metric-direct','metric-neighbors','metric-top1','metric-perfect'])$(id).textContent='—';
  $('metric-piece-count').textContent='Correct pieces / total pieces';
  $('run-label').textContent=meta.models.find(m=>m.id===s.model).label.toUpperCase();
  $('puzzle-label').textContent=s.source?'CUSTOM IMAGE':`IMAGE ${String(s.index+1).padStart(3,'0')}`;
  $('dimensions').textContent='— PIECES';$('stage').textContent='Preparing';$('scrubber').value=0;$('step-count').textContent='0 / 0';
  $('anchor-legend').hidden=!s.anchor&&!free;
  $('seed-legend-label').textContent=free?'Seed pair':'Anchor';
  $('result-context').textContent='Preparing final-puzzle metrics. Playback will show recorded solver states.';
  setLoading('Loading validation pieces…');
  const url='/api/puzzle?'+new URLSearchParams(s);
  try{
    let result;
    while(ticket===request){
      result=await fetchJSON(url);if(ticket!==request)return;
      if(result.status==='error')throw new Error(result.message);
      if(result.status==='ready')break;
      $('loading-message').textContent=result.message;await delay(1200);
    }
    if(ticket!==request)return;
    const puzzle=await fetchJSON(result.url);
    const [image,reference]=await Promise.all([loadImage(`/cache/${puzzle.key}/pieces.png`),loadImage(`/cache/${puzzle.key}/reference.png`)]);
    if(ticket!==request)return;
    data=puzzle;atlas=image;frame=0;wrongSlots=placementErrors(puzzle);
    $('reference').src=reference.src;$('reference').hidden=false;
    $('source-name').textContent=s.source?customName:puzzle.image.split('/').pop();$('source-index').textContent=s.source?'Custom experiment · square center crop':`Validation index ${puzzle.val_index} · image ${s.index+1} of 100`;
    $('dimensions').textContent=`${puzzle.rows} × ${puzzle.cols}${free?' BOUNDS':''} / ${puzzle.piece_count} PIECES`;
    $('scrubber').max=puzzle.frames.length-1;
    $('metric-direct').textContent=pct(puzzle.metrics.direct);$('metric-neighbors').textContent=pct(puzzle.metrics.neighbors);$('metric-top1').textContent=pct(puzzle.metrics.top1);
    const correctPieces=puzzle.metrics.correct_count??puzzle.frames.at(-1).grid.filter((id,i)=>id===puzzle.truth[i]).length;
    $('metric-piece-count').textContent=`${correctPieces} / ${puzzle.piece_count} correct`;
    $('metric-perfect').textContent=puzzle.metrics.perfect?'Complete':'Incomplete';
    if(free){$('free-placed').textContent=`${puzzle.metrics.placed} / ${puzzle.piece_count}`;$('free-recall').textContent=pct(puzzle.metrics.neighbor_recall);$('free-bounds').textContent=`${puzzle.cols} × ${puzzle.rows}`;}
    $('result-context').textContent=`Final-puzzle metrics · ${s.strategy==='shrink'?`shrinking frame · translation aligned · growth alone ${pct(puzzle.metrics.growth_direct)} → frame ${pct(puzzle.metrics.before_polish_direct)} → polish ${pct(puzzle.metrics.direct)} · ${puzzle.metrics.evicted} moves, ${puzzle.metrics.swaps} swaps`:free?'strongest pair · translation aligned':s.anchor?'top-left anchored':'unanchored'} · ${s.source?'custom image (not held-out validation)':`image ${s.index+1} of 100`}${!s.source&&s.category.startsWith('flat_')&&s.index>=30?' · beyond the original 30-puzzle flat-cut comparison':''}`;
    $('loading').hidden=true;document.querySelector('.viewer').setAttribute('aria-busy','false');controls(true);draw();
    refreshUpdater();
    refreshFailures();
    $('announcement').textContent=`${s.source?'Custom image':`Image ${s.index+1}`} loaded. ${puzzle.piece_count} pieces. ${puzzle.metrics.perfect?'Final puzzle solved.':'Final puzzle incomplete.'}`;
    if(!matchMedia('(prefers-reduced-motion: reduce)').matches)play();
  }catch(error){if(ticket!==request)return;document.querySelector('.spinner').hidden=true;$('loading-title').textContent='Couldn’t prepare this puzzle';$('loading-message').textContent=error.message;$('retry').hidden=false;document.querySelector('.viewer').setAttribute('aria-busy','false');$('announcement').textContent=error.message;}
}
function sprite(ctx,id,x,y,size,alpha=1){const {box,atlas_cols}=data;ctx.globalAlpha=alpha;ctx.drawImage(atlas,(id%atlas_cols)*box,Math.floor(id/atlas_cols)*box,box,box,x,y,size,size);ctx.globalAlpha=1;}
function draw(){
  if(!data)return;
  const current=data.frames[frame], {rows,cols,cell,pad,box}=data;
  const canvas=$('board'),ctx=canvas.getContext('2d');
  const margin=28, scale=(canvas.width-2*margin)/Math.max(cols*cell,rows*cell);
  const ox=(canvas.width-cols*cell*scale)/2,oy=(canvas.height-rows*cell*scale)/2;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f2f0e9';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#deded2';ctx.lineWidth=1;
  for(let x=0;x<=cols;x++){ctx.beginPath();ctx.moveTo(ox+x*cell*scale,oy);ctx.lineTo(ox+x*cell*scale,oy+rows*cell*scale);ctx.stroke();}
  for(let y=0;y<=rows;y++){ctx.beginPath();ctx.moveTo(ox,oy+y*cell*scale);ctx.lineTo(ox+cols*cell*scale,oy+y*cell*scale);ctx.stroke();}
  // Proposals first, committed pieces above them, mirroring animate_solve.py.
  ctx.save();ctx.beginPath();ctx.rect(ox,oy,cols*cell*scale,rows*cell*scale);ctx.clip();
  for(const locked of [false,true])current.grid.forEach((id,i)=>{if(id<0||current.locked[i]!==locked)return;const x=ox+((i%cols)*cell-pad)*scale,y=oy+(Math.floor(i/cols)*cell-pad)*scale;sprite(ctx,id,x,y,box*scale,locked?1:.46);});
  ctx.restore();
  if(data.anchored&&frame>0){ctx.strokeStyle='#b57b60';ctx.lineWidth=3;ctx.strokeRect(ox+1.5,oy+1.5,cell*scale-3,cell*scale-3);}
  if(current.frame){const [fx0,fy0,fx1,fy1]=current.frame;ctx.save();ctx.setLineDash([10,6]);ctx.strokeStyle='#3d6f8e';ctx.lineWidth=3;ctx.strokeRect(ox+fx0*cell*scale,oy+fy0*cell*scale,(fx1-fx0+1)*cell*scale,(fy1-fy0+1)*cell*scale);ctx.restore();}
  if(isFree(data.strategy)&&frame>0&&!current.frame){ctx.strokeStyle='#b57b60';ctx.lineWidth=3;for(const i of data.seed.slots)ctx.strokeRect(ox+(i%cols)*cell*scale+1.5,oy+Math.floor(i/cols)*cell*scale+1.5,cell*scale-3,cell*scale-3);}
  const finalFrame=frame===data.frames.length-1;
  const wrongCount=wrongSlots.filter(Boolean).length;
  $('incorrect-legend').hidden=!finalFrame;
  $('incorrect-count').textContent=wrongCount?`${wrongCount} misplaced`:'No misplaced pieces';
  if(finalFrame&&$('show-errors').checked){
    ctx.fillStyle='rgba(197,69,66,.19)';ctx.strokeStyle='#bd5551';ctx.lineWidth=2;
    wrongSlots.forEach((wrong,i)=>{if(!wrong)return;const x=ox+(i%cols)*cell*scale,y=oy+Math.floor(i/cols)*cell*scale;ctx.fillRect(x,y,cell*scale,cell*scale);ctx.strokeRect(x+1,y+1,cell*scale-2,cell*scale-2);});
  }
  const set=new Set(current.grid.filter((id,i)=>id>=0&&current.locked[i]));
  const tray=$('tray'),tc=tray.getContext('2d');tc.clearRect(0,0,tray.width,tray.height);
  const count=data.piece_count, n=Math.ceil(Math.sqrt(count)),tile=tray.width/n;
  for(let i=0;i<count;i++)sprite(tc,i,(i%n)*tile+2,Math.floor(i/n)*tile+2,tile-4,set.has(i)?.12:1);
  $('remaining').textContent=`${count-set.size} remaining`;
  $('stage').textContent=frame===0?'Ready to assemble':current.label==='stage 4'?'Final assignment':current.label.replace('stage 2','Initial placement').replace('stage 3','Refinement');
  $('scrubber').value=frame;$('step-count').textContent=`${frame} / ${data.frames.length-1}`;
  const free=isFree(data.strategy);
  $('live-updater').hidden=free;
  if(!free){
    const live=snapshotMetrics(data,current);
    $('metric-direct').textContent=pct(live.accuracy);$('metric-piece-count').textContent=`${live.correct} / ${data.piece_count} correct`;
    $('metric-neighbors').textContent=pct(live.neighbors);$('metric-perfect').textContent=live.perfect?'Complete':live.filled<data.piece_count?'In progress':'Incomplete';
    $('live-correct').textContent=`${live.correct} / ${data.piece_count}`;
    const initial=data.frames.filter(f=>f.label==='stage 2').at(-1);
    const baseline=initial?snapshotMetrics(data,initial).accuracy:0;
    const delta=(live.accuracy-baseline)*100;
    $('live-delta').textContent=live.filled===data.piece_count?`${delta>=0?'+':''}${delta.toFixed(1)} pp`:'—';
    $('live-wrong-locks').textContent=live.wrongLocks;
    $('location-label').textContent='PLACEMENT ACCURACY';
    $('result-context').textContent=`Metrics follow playback · ${current.label} · ${data.anchored?'top-left anchored':'unanchored'} · ${data.mode==='custom'?'custom image':`image ${data.index+1} of 100`}`;
  }
  drawUpdaterCursor();
}

async function refreshUpdater(){
  const ticket=++updaterRequest;
  if(!data||mode!=='validation'||isFree(data.strategy)){$('updater-analysis').hidden=true;return;}
  $('updater-analysis').hidden=false;
  const s=selection();
  try{
    const result=await fetchJSON('/api/updater?'+new URLSearchParams({model:s.model,category:s.category,anchor:s.anchor}));
    if(ticket!==updaterRequest)return;
    updaterData=result;
    $('updater-coverage').textContent=`${result.evaluated} / 100 puzzles evaluated`;
    $('updater-table').replaceChildren();
    for(const row of result.rows){const tr=document.createElement('tr');for(const value of [row.label,`${row.solved}/${result.evaluated}`,pct(row.solve_rate),pct(row.accuracy),pct(row.neighbors),row.wrong_locks]){const td=document.createElement('td');td.textContent=value;tr.append(td);}$('updater-table').append(tr);}
    $('updater-effect').hidden=!result.evaluated;
    $('updater-rescued').textContent=result.rescued??0;
    $('updater-broken').textContent=result.broken??0;
    renderUpdaterChart();drawUpdaterCursor();
  }catch(error){if(ticket===updaterRequest)$('updater-coverage').textContent=`Could not load updater analysis: ${error.message}`;}
}
function renderUpdaterChart(){
  const svg=$('updater-chart'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
  if(!updaterData?.rows.length)return;
  const rows=updaterData.rows,x=i=>45+i*620/Math.max(1,rows.length-1),y=v=>140-120*v;
  const add=(tag,attrs,text)=>{const element=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))element.setAttribute(k,v);if(text!==undefined)element.textContent=text;svg.append(element);return element;};
  for(const value of [0,.5,1]){add('line',{x1:45,y1:y(value),x2:665,y2:y(value),stroke:'#dedfd4'});add('text',{x:35,y:y(value)+4,'text-anchor':'end',fill:'#818479','font-size':10},`${value*100}%`);}
  for(const [key,color] of [['solve_rate','#657859'],['accuracy','#b57b60']])add('polyline',{points:rows.map((r,i)=>`${x(i)},${y(r[key])}`).join(' '),fill:'none',stroke:color,'stroke-width':2});
  const ticks=new Set([0,rows.length-1,...rows.map((_,i)=>i).filter(i=>i%Math.max(1,Math.ceil(rows.length/7))===0)]);
  for(const i of ticks)add('text',{x:x(i),y:164,'text-anchor':'middle',fill:'#818479','font-size':9},i===0?'Initial':i===rows.length-1?'Final':`R${i}`);
  add('line',{id:'updater-cursor',x1:45,x2:45,y1:15,y2:145,stroke:'#343b32','stroke-dasharray':'3 3'});
}
function drawUpdaterCursor(){
  if(!data||!updaterData?.rows.length||isFree(data.strategy))return;
  const label=data.frames[frame].label;
  const key=label==='stage 4'?'final':label.startsWith('stage 3')?`round-${label.match(/round (\d+)/)?.[1]||1}`:'initial';
  const i=Math.max(0,updaterData.rows.findIndex(r=>r.key===key)),row=updaterData.rows[i];
  const cursor=$('updater-cursor');if(cursor){const x=45+i*620/Math.max(1,updaterData.rows.length-1);cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);}
  $('updater-pass').textContent=row.label;
  $('updater-rate').textContent=pct(row.solve_rate);
  $('updater-solved').textContent=`${row.solved} / ${updaterData.evaluated} solved`;
  $('updater-secondary').textContent=`${pct(row.accuracy)} mean placement · ${row.wrong_locks} wrong locks`;
}
function play(){
  if(!data)return;
  if(frame>=data.frames.length-1)frame=0;
  $('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Pause assembly');
  const tick=()=>{if(!data)return;if(frame>=data.frames.length-1){stop();return;}frame++;draw();const hold=data.frames[frame].label.startsWith('stage 3')||data.frames[frame].label==='Strongest pair';timer=setTimeout(tick,(hold?650:110)/Number($('speed').value));};
  timer=setTimeout(tick,450/Number($('speed').value));
}
$('play').onclick=()=>timer?stop():play();
$('restart').onclick=()=>{stop();frame=0;draw();};
$('step-back').onclick=()=>{stop();frame=Math.max(0,frame-1);draw();};
$('step-forward').onclick=()=>{stop();frame=Math.min(data.frames.length-1,frame+1);draw();};
$('scrubber').oninput=()=>{stop();frame=Number($('scrubber').value);draw();};
$('show-errors').onchange=draw;
$('load').onclick=load;$('retry').onclick=load;
$('image-number').onkeydown=e=>{if(e.key==='Enter')load();};
for(const [id,delta] of [['previous-image',-1],['next-image',1]])$(id).onclick=()=>{$('image-number').value=Math.max(1,Math.min(100,Number($('image-number').value)+delta));load();};
$('model').onchange=load;
$('cut-type').onchange=()=>setCategory($('cut-type').value);
$('grid-size').onchange=load;
async function setMode(next){
  mode=next;
  for(const value of ['validation','custom']){$(`mode-${value}`).classList.toggle('active',mode===value);$(`mode-${value}`).setAttribute('aria-pressed',String(mode===value));}
  for(const id of ['custom-source','custom-settings'])$(id).hidden=mode!=='custom';
  $('validation-picker').hidden=mode==='custom';
  document.querySelector('.nav-heading').hidden=mode==='custom';
  $('categories').hidden=mode==='custom';
  $('failure-toggle').hidden=mode==='custom';
  $('failure-panel').hidden=mode==='custom'||!failureOpen;
  $('cohort-label').textContent=mode==='custom'?'Custom image':'SA-1B · validation';
  if(mode==='custom'&&!['clean','flat_even','flat_diagonal'].includes(category))category='clean';
  setCategory(category);
}
$('mode-validation').onclick=()=>setMode('validation');
$('mode-custom').onclick=()=>setMode('custom');
$('upload-button').onclick=()=>$('image-upload').click();
$('image-upload').onchange=async()=>{
  const file=$('image-upload').files[0];if(!file)return;
  if(file.size>12*1024*1024){$('upload-status').textContent='Choose an image smaller than 12 MB.';return;}
  uploading=true;$('load').disabled=true;stop();++request;controls(false);
  $('upload-status').textContent='Preparing image…';
  try{
    const response=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file});
    const result=await response.json();if(!response.ok)throw new Error(result.message);
    customSource=result.id;customName=file.name;$('upload-name').textContent=file.name;
    document.querySelectorAll('.gallery button').forEach(b=>{b.classList.remove('selected');b.setAttribute('aria-pressed','false');});
    $('upload-status').textContent='Image ready · square crop';
    uploading=false;await load();
  }catch(error){$('upload-status').textContent=error.message;if(data)controls(true);}
  finally{uploading=false;$('load').disabled=false;$('image-upload').value='';}
};
async function initGallery(){
  try{const images=await fetchJSON('/api/gallery');images.forEach((item,i)=>{const button=document.createElement('button');button.className=i===0?'selected':'';button.setAttribute('aria-label',`Use gallery image ${i+1}: ${item.label}`);button.setAttribute('aria-pressed',String(i===0));const image=document.createElement('img');image.src=item.thumbnail;image.alt=item.label;image.loading='lazy';button.append(image);button.onclick=()=>{if(uploading)return;customSource=item.id;customName=item.label;$('upload-name').textContent=item.label;$('upload-status').textContent='Square crop · stays local';document.querySelectorAll('.gallery button').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button));});load();};$('gallery').append(button);});}catch(error){$('upload-status').textContent=`Gallery unavailable: ${error.message}. You can still upload an image.`;}
}
initGallery();
async function refreshFailures(){
  const ticket=++failureRequest;
  if(mode!=='validation')return;
  const s=selection();
  $('failure-count').textContent='…';$('failure-coverage').textContent='Loading evaluated results…';
  $('failure-list').replaceChildren();$('next-failure').disabled=true;failures=[];
  try{
    const result=await fetchJSON('/api/results?'+new URLSearchParams({model:s.model,category:s.category,anchor:s.anchor,strategy:s.strategy}));
    if(ticket!==failureRequest||mode!=='validation')return;
    failures=result.failures;
    $('failure-count').textContent=`${result.failed} / ${result.evaluated} tested`;
    $('failure-coverage').textContent=`${result.evaluated} / ${result.total} tested · ${result.solved} solved`;
    $('next-failure').disabled=!failures.length;
    if(!failures.length){const empty=document.createElement('p');empty.className='failure-empty';empty.textContent=result.evaluated?'No failures among evaluated images.':'No evaluated results yet. Load a validation image to record its result.';$('failure-list').append(empty);}
    for(const item of failures){
      const button=document.createElement('button');button.className='failure-case';
      button.classList.toggle('current',item.index===Number($('image-number').value)-1);
      button.setAttribute('aria-label',`Open unsolved image ${item.number}, ${pct(item.accuracy)} correctly placed`);
      const name=document.createElement('strong');name.textContent=`Image ${String(item.number).padStart(3,'0')}`;
      const accuracy=document.createElement('span');accuracy.textContent=`${pct(item.accuracy)} correct ↗`;
      button.append(name,accuracy);button.onclick=()=>{$('image-number').value=item.number;load();};$('failure-list').append(button);
    }
  }catch(error){if(ticket!==failureRequest)return;$('failure-count').textContent='—';$('failure-coverage').textContent=`Could not load results: ${error.message}`;}
}
$('failure-toggle').onclick=()=>{failureOpen=!failureOpen;$('failure-panel').hidden=!failureOpen;$('failure-toggle').setAttribute('aria-expanded',String(failureOpen));if(failureOpen)refreshFailures();};
$('next-failure').onclick=()=>{if(!failures.length)return;const current=Number($('image-number').value)-1;const next=failures.find(f=>f.index>current)||failures[0];$('image-number').value=next.number;load();};
$('assembly-mode').onchange=()=>{const value=$('assembly-mode').value;anchored=value==='anchored';strategy=isFree(value)?value:'grid';anchorUI();load();};
$('about-toggle').onclick=()=>{const open=$('about').hidden;$('about').hidden=!open;$('about-toggle').setAttribute('aria-expanded',String(open));};
document.addEventListener('click',event=>{
  const trigger=event.target.closest('.help-trigger');
  const clicked=trigger?.closest('.help');
  document.querySelectorAll('.help').forEach(help=>{
    const open=help===clicked&&!help.classList.contains('is-open');
    help.classList.toggle('is-open',open);help.querySelector('button').setAttribute('aria-expanded',String(open));
  });
});
document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelectorAll('.help.is-open').forEach(help=>{help.classList.remove('is-open');help.querySelector('button').setAttribute('aria-expanded','false');help.querySelector('button').blur();});});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName)||!data)return;if(e.code==='Space'){e.preventDefault();timer?stop():play();}if(e.code==='ArrowLeft'){stop();frame=Math.max(0,frame-1);draw();}if(e.code==='ArrowRight'){stop();frame=Math.min(data.frames.length-1,frame+1);draw();}});
controls(false);
async function refreshBatch(){
  try{
    const state=await fetchJSON('/api/batch-status');
    $('batch-status').hidden=state.status==='idle';
    if(state.status==='running'){
      const current=state.current;
      $('batch-status').textContent=`Caching validation · ${state.completed.toLocaleString()} / ${state.total.toLocaleString()}`;
      $('batch-status').title=current?`Current: image ${current.index+1}, ${current.model}, ${current.category}`:'';
      $('batch-status').style.setProperty('--cache-progress',`${100*state.completed/state.total}%`);
      if(meta&&mode==='validation')refreshFailures();
      if(data&&mode==='validation'&&!isFree(data.strategy))refreshUpdater();
    }else if(state.status==='complete')$('batch-status').textContent=`Validation complete · ${state.completed.toLocaleString()} replays`;
    else if(state.status!=='idle')$('batch-status').textContent=`Validation cache ${state.status} · ${state.completed??0} / ${state.total??'—'} ready. ${state.message||'Run precompute.py to resume.'}`;
    if(state.status==='complete'&&localStorage.getItem('interlock-notified-batch')!==state.started){
      localStorage.setItem('interlock-notified-batch',state.started);
      document.title='✓ Validation complete — interlock';
      $('announcement').textContent=`Validation completed. ${state.completed} of ${state.total} replays cached.`;
      if('Notification' in window&&Notification.permission==='granted')new Notification('Interlock validation complete',{body:`${state.completed} / ${state.total} replays are cached. Open the viewer to compare results.`});
      if(data&&mode==='validation'&&!isFree(data.strategy))refreshUpdater();
    }
  }catch(error){$('batch-status').hidden=true;}
}
$('notify-batch').onclick=async()=>{
  if(!('Notification' in window)){$('notify-batch').textContent='Completion shown in page';return;}
  try{const permission=await Notification.requestPermission();$('notify-batch').textContent=permission==='granted'?'Notifications enabled':'Completion shown in page';}catch(error){$('notify-batch').textContent='Completion shown in page';}
};
refreshBatch();setInterval(refreshBatch,10000);
(async()=>{try{meta=await fetchJSON('/api/meta');for(const model of meta.models){const option=document.createElement('option');option.value=model.id;option.textContent=model.label;$('model').append(option);}meta.categories.forEach((c,i)=>{const b=document.createElement('button');b.className='category';b.dataset.id=c.id;b.innerHTML=`<span class="nav-icon" aria-hidden="true">${icons[i]}</span><span>${c.label}</span><span class="nav-arrow" aria-hidden="true">↗</span>`;b.onclick=()=>setCategory(c.id);$('categories').append(b);});setCategory(category);}catch(error){$('loading-title').textContent='Start the local viewer server';$('loading-message').textContent=error.message;document.querySelector('.spinner').hidden=true;}})();
