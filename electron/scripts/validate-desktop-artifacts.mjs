import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const electronDir = path.resolve(__dirname, '..');
const rootDir = path.resolve(electronDir, '..');

const forbiddenRuntimeRoots = new Set([
  'data',
  'uploads',
  'logs',
  'cache',
  'history',
  'memory',
  'vector_stores',
  'semantic_groups',
  'overviews',
  'parse',
]);

function fail(message) {
  console.error(`[validate-desktop-artifacts] ${message}`);
  process.exit(1);
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    fail(`Cannot read JSON: ${path.relative(rootDir, filePath)} (${error.message})`);
  }
}

function requireDir(dirPath) {
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    fail(`Missing directory: ${path.relative(rootDir, dirPath)}`);
  }
}

function findFile(dirPath, filename, maxDepth = 3, depth = 0) {
  if (!fs.existsSync(dirPath) || depth > maxDepth) {
    return null;
  }
  for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isFile() && entry.name === filename) {
      return fullPath;
    }
    if (entry.isDirectory()) {
      const found = findFile(fullPath, filename, maxDepth, depth + 1);
      if (found) {
        return found;
      }
    }
  }
  return null;
}

function compareField(label, left, right, field, required = true) {
  const leftValue = left[field] ?? '';
  const rightValue = right[field] ?? '';
  if (!required && (!leftValue || !rightValue)) {
    return;
  }
  if (leftValue !== rightValue) {
    fail(`${label} ${field} mismatch: ${leftValue || '<empty>'} != ${rightValue || '<empty>'}`);
  }
}

function scanForPrivateFiles(dirPath, label) {
  const stack = [dirPath];
  while (stack.length) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      const rel = path.relative(dirPath, fullPath).replaceAll(path.sep, '/');
      const parts = rel.split('/').map((part) => part.toLowerCase());
      const first = parts[0] || '';
      const second = parts[1] || '';
      const lowerName = entry.name.toLowerCase();

      if (
        forbiddenRuntimeRoots.has(first) ||
        (first === '_internal' && forbiddenRuntimeRoots.has(second)) ||
        lowerName === '.env' ||
        lowerName.startsWith('.env.') ||
        lowerName.endsWith('.log') ||
        lowerName.includes('api_key') ||
        lowerName.includes('apikey')
      ) {
        fail(`${label} contains private/runtime file: ${rel}`);
      }

      if (entry.isDirectory()) {
        stack.push(fullPath);
      }
    }
  }
}

const versionJson = readJson(path.join(rootDir, 'version.json'));
const rootBuild = readJson(path.join(rootDir, 'build-info.json'));
const frontendBuildDir = path.join(rootDir, 'frontend', 'build');
const frontendBuild = readJson(path.join(frontendBuildDir, 'build-info.json'));
const backendSourceBuild = readJson(path.join(rootDir, 'backend', 'build-info.json'));
const backendDistDir = path.join(rootDir, 'backend', 'dist', 'chatpdf-backend');
const backendDistBuildPath = findFile(backendDistDir, 'build-info.json');

requireDir(frontendBuildDir);
requireDir(backendDistDir);
if (!backendDistBuildPath) {
  fail('Frozen backend is missing build-info.json; rebuild with scripts/build-all.bat');
}

const backendDistBuild = readJson(backendDistBuildPath);
const electronPkg = readJson(path.join(electronDir, 'package.json'));

compareField('version.json vs build-info', versionJson, rootBuild, 'version');
compareField('electron/package.json vs build-info', electronPkg, rootBuild, 'version');
for (const [label, manifest] of [
  ['frontend/build', frontendBuild],
  ['backend/build-info', backendSourceBuild],
  ['backend/dist', backendDistBuild],
]) {
  compareField(`${label} vs root build-info`, manifest, rootBuild, 'version');
  compareField(`${label} vs root build-info`, manifest, rootBuild, 'git_sha', false);
  compareField(`${label} vs root build-info`, manifest, rootBuild, 'build_time', false);
}

scanForPrivateFiles(frontendBuildDir, 'frontend build');
scanForPrivateFiles(backendDistDir, 'backend dist');

console.log(
  `[validate-desktop-artifacts] OK v${rootBuild.version} ${rootBuild.git_short_sha || 'no-git'} dirty=${rootBuild.build_dirty}`,
);
