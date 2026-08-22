#!/usr/bin/env node
/**
 * Convert a 3DGS PLY to SPZ (v2, gzipped — mkkellogg-compatible).
 *
 * usage:  node ply_to_spz.mjs <input.ply> <output.spz>
 *
 * Uses the vendored spz-js library; no external npm install needed.
 */
import { createReadStream, writeFileSync, statSync } from "node:fs";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

// Import the sub-modules directly so we don't trip the `playcanvas`
// dep that the `compressed-ply-*` modules pull in (we don't need them).
const { loadPly }       = await import(resolve(__dirname, "spz-js/dist/ply-loader.js"));
const { serializeSpz }  = await import(resolve(__dirname, "spz-js/dist/spz-serializer.js"));

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
    console.error("usage: node ply_to_spz.mjs <input.ply> <output.spz>");
    process.exit(2);
}

const t0 = Date.now();
const stream = createReadStream(inPath);
const web    = Readable.toWeb(stream);
const gs     = await loadPly(web);
const tLoad  = (Date.now() - t0) / 1000;

const t1     = Date.now();
const buf    = await serializeSpz(gs);
const tSer   = (Date.now() - t1) / 1000;

writeFileSync(outPath, Buffer.from(buf));

const inMB  = statSync(inPath).size  / (1024 * 1024);
const outMB = statSync(outPath).size / (1024 * 1024);
console.log(JSON.stringify({
    numPoints: gs.numPoints,
    plyMB: +inMB.toFixed(2),
    spzMB: +outMB.toFixed(2),
    ratio: +(inMB / outMB).toFixed(2),
    loadSec: +tLoad.toFixed(2),
    serSec:  +tSer.toFixed(2),
}));
