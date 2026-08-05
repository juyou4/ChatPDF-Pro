import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const electronDir = path.resolve(__dirname, '..');
const rootDir = path.resolve(electronDir, '..');
const releaseDir = path.join(electronDir, 'release');

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sha256(filePath) {
  const hash = crypto.createHash('sha256');
  hash.update(fs.readFileSync(filePath));
  return hash.digest('hex').toUpperCase();
}

function writeJson(filePath, payload) {
  fs.writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

if (!fs.existsSync(releaseDir)) {
  console.log('[stamp-release-artifacts] release directory does not exist, skipped');
  process.exit(0);
}

const buildInfo = readJson(path.join(rootDir, 'build-info.json'));
const pkg = readJson(path.join(electronDir, 'package.json'));
const version = String(pkg.version || buildInfo.version || '').trim();
const shortSha = String(buildInfo.git_short_sha || 'nogit').trim() || 'nogit';
const dirtySuffix = buildInfo.build_dirty ? '-dirty' : '';
const stamp = `${version}-${shortSha}${dirtySuffix}`;
const stampableExts = new Set(['.exe', '.dmg', '.appimage', '.zip', '.msi']);
const manifest = {
  schema_version: 1,
  product: pkg.productName || pkg.name,
  version,
  stamp,
  build: buildInfo,
  artifacts: [],
};

for (const entry of fs.readdirSync(releaseDir, { withFileTypes: true })) {
  if (!entry.isFile()) {
    continue;
  }
  const ext = path.extname(entry.name).toLowerCase();
  const isBlockmap = entry.name.toLowerCase().endsWith('.blockmap');
  if (!stampableExts.has(ext) && !isBlockmap) {
    continue;
  }
  if (!entry.name.includes(version) || entry.name.includes(stamp)) {
    continue;
  }

  const oldPath = path.join(releaseDir, entry.name);
  const newName = entry.name.replace(version, stamp);
  const newPath = path.join(releaseDir, newName);
  if (fs.existsSync(newPath)) {
    fs.rmSync(newPath, { force: true });
  }
  fs.renameSync(oldPath, newPath);

  if (!isBlockmap) {
    const digest = sha256(newPath);
    fs.writeFileSync(`${newPath}.sha256`, `${digest}  ${newName}\n`, 'utf8');
    manifest.artifacts.push({
      file: newName,
      size: fs.statSync(newPath).size,
      sha256: digest,
    });
  }

  const latestPath = path.join(releaseDir, 'latest.yml');
  if (fs.existsSync(latestPath)) {
    const before = fs.readFileSync(latestPath, 'utf8');
    fs.writeFileSync(latestPath, before.split(entry.name).join(newName), 'utf8');
  }
}

writeJson(path.join(releaseDir, 'release-manifest.json'), manifest);
console.log(`[stamp-release-artifacts] wrote release-manifest.json (${manifest.artifacts.length} artifact(s))`);
