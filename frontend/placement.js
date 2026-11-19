/* Evaluation-only overlays. Strongest-pair coordinates use the recorded best global translation. */
function placementErrors(puzzle) {
  const final = puzzle.frames[puzzle.frames.length - 1];
  return final.grid.map((id, slot) => {
    if (id < 0) return false;
    if (puzzle.strategy !== 'strongest' && puzzle.strategy !== 'shrink') return id !== puzzle.truth[slot];
    const point = puzzle.coordinates[id], truth = puzzle.true_positions[id];
    const shift = puzzle.metrics.alignment_shift;
    return point[0] + shift[0] !== truth[0] || point[1] + shift[1] !== truth[1];
  });
}
if (typeof module !== 'undefined') module.exports = {placementErrors};

function snapshotMetrics(puzzle, frame) {
  const n=puzzle.piece_count, grid=frame.grid;
  let correct=0,filled=0,wrongLocks=0,joins=0;
  const positions=[];
  puzzle.truth.forEach((id,i)=>{if(id>=0)positions[id]=[i%puzzle.cols,Math.floor(i/puzzle.cols)];});
  const prior=puzzle.frames.filter(f=>f.label.startsWith('stage 3'));
  const locked=frame.label==='stage 4'?(prior.at(-1)?.locked||grid.map(()=>false)):frame.locked;
  grid.forEach((id,i)=>{
    if(id<0)return;filled++;if(id===puzzle.truth[i])correct++;else if(locked[i])wrongLocks++;
    for(const [other,dx,dy] of [[i+1,1,0],[i+puzzle.cols,0,1]]){
      if(dx&&i%puzzle.cols===puzzle.cols-1)continue;
      if(other<grid.length&&grid[other]>=0){const a=positions[id],b=positions[grid[other]];if(b[0]-a[0]===dx&&b[1]-a[1]===dy)joins++;}
    }
  });
  const total=puzzle.rows*(puzzle.cols-1)+(puzzle.rows-1)*puzzle.cols;
  return {correct,filled,accuracy:correct/n,neighbors:total?joins/total:0,perfect:correct===n&&filled===n,wrongLocks};
}
if (typeof module !== 'undefined') module.exports.snapshotMetrics=snapshotMetrics;
