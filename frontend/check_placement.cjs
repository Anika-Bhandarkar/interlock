const assert=require('node:assert/strict');
const {placementErrors}=require('./placement.js');
assert.deepEqual(placementErrors({frames:[{grid:[0,2,1,3]}],truth:[0,1,2,3]}),[false,true,true,false]);
assert.deepEqual(placementErrors({frames:[{grid:[-1,0,1]}],truth:[-1,0,2]}),[false,false,true]);
const free={strategy:'strongest',frames:[{grid:[0,1,-1,2,3]}],
  coordinates:[[10,20],[11,20],[10,21],[12,21]],
  true_positions:[[0,0],[1,0],[0,1],[1,1]],metrics:{alignment_shift:[-10,-20]}};
assert.deepEqual(placementErrors(free),[false,false,false,false,true]);
assert.equal(placementErrors(free).filter(Boolean).length,1);
console.log('Placement overlays verified for fixed grids, empty cells, and translation-aligned free assemblies.');
const {snapshotMetrics}=require('./placement.js');
const puzzle={piece_count:4,rows:2,cols:2,truth:[0,1,2,3],frames:[
 {label:'stage 3, round 1',grid:[1,0,2,3],locked:[true,false,false,false]},
 {label:'stage 4',grid:[1,0,2,3],locked:[true,true,true,true]}]};
assert.equal(snapshotMetrics(puzzle,puzzle.frames[1]).accuracy,.5);
assert.equal(snapshotMetrics(puzzle,puzzle.frames[1]).wrongLocks,1);
assert.equal(snapshotMetrics(puzzle,{label:'stage 2',grid:[0,1,-1,-1],locked:[false,false,false,false]}).neighbors,.25);
console.log('Playback accuracy and updater lock metrics verified.');
