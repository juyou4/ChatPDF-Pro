import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const electronDir = path.resolve(__dirname, '..');
const rootDir = path.resolve(electronDir, '..');

// These names are private only when they are artifact-level runtime roots.
// Dependency packages legitimately contain paths such as `torch/utils/data`
// and `matplotlib/mpl-data`; matching every path segment would reject a
// clean, reproducible build.
const privateRuntimeRoots = new Set([
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

// Unlike runtime roots, development/evaluation material is never a runtime
// dependency and can be rejected wherever it appears in the artifact.
const forbiddenDevelopmentRoots = new Set([
  // Development fixtures and evaluation material are never runtime assets.
  'test',
  'tests',
  'fixture',
  'fixtures',
  'course',
  'courses',
  'eval',
  'evaluation',
  'evaluations',
  '课程',
]);

const privateExtensions = new Set([
  '.pdf',
  '.db',
  '.sqlite',
  '.sqlite3',
  '.faiss',
  '.pkl',
  '.pickle',
  '.log',
]);

const forbiddenBasenames = new Set([
  'online_ocr_config.json',
  'ocr_provider_usage.json',
  'chat_history.json',
  'history.json',
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

function currentGitState() {
  try {
    const sha = execFileSync('git', ['rev-parse', '--verify', 'HEAD'], {
      cwd: rootDir,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    const status = execFileSync('git', ['status', '--porcelain', '--untracked-files=normal'], {
      cwd: rootDir,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return { sha, dirty: Boolean(status.trim()) };
  } catch {
    return { sha: '', dirty: null };
  }
}

function gitText(args) {
  return execFileSync('git', args, {
    cwd: rootDir,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  });
}

function sourceFingerprint() {
  try {
    const headSha = gitText(['rev-parse', '--verify', 'HEAD']).trim();
    if (!headSha) return '';
    const tracked = gitText(['diff', '--name-only', '-z', 'HEAD']);
    const untracked = gitText(['ls-files', '--others', '--exclude-standard', '-z']);
    const paths = new Set(
      `${tracked}${untracked}`
        .split('\0')
        .filter(Boolean)
        .map((value) => value.replaceAll('\\', '/')),
    );
    const digest = crypto.createHash('sha256');
    digest.update(`${headSha}\0`);
    for (const relative of [...paths].sort()) {
      digest.update(`${relative}\0`);
      const fullPath = path.join(rootDir, ...relative.split('/'));
      if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
        digest.update('file\0');
        digest.update(sha256(fullPath).toLowerCase());
      } else {
        digest.update('deleted\0');
      }
      digest.update('\0');
    }
    return digest.digest('hex');
  } catch {
    return '';
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
      const lowerName = entry.name.toLowerCase();
      const extension = path.extname(lowerName);

      // A runtime root may be emitted directly by Electron/PyInstaller or
      // below their conventional wrapper directory. It must not match an
      // arbitrary dependency segment such as `torch/utils/data`.
      const rootParts = parts[0] === '_internal' ? parts.slice(1) : parts;
      const runtimeRoot = rootParts[0];
      const isPrivateRuntimeRoot = privateRuntimeRoots.has(runtimeRoot);
      const isDevelopmentRoot = parts.some((part) => forbiddenDevelopmentRoots.has(part));
      const isSensitiveRootFile = rootParts.length <= 1 && privateExtensions.has(extension);

      if (
        isPrivateRuntimeRoot ||
        isDevelopmentRoot ||
        isSensitiveRootFile ||
        forbiddenBasenames.has(lowerName) ||
        lowerName === '.env' ||
        lowerName.startsWith('.env.') ||
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
const frontendBuildDir = process.env.CHATPDF_VALIDATE_FRONTEND_DIR || path.join(rootDir, 'frontend', 'build');
const frontendBuild = readJson(path.join(frontendBuildDir, 'build-info.json'));
const backendSourceBuild = readJson(path.join(rootDir, 'backend', 'build-info.json'));
const backendDistDir = process.env.CHATPDF_VALIDATE_BACKEND_DIR || path.join(rootDir, 'backend', 'dist', 'chatpdf-backend');
const backendDistBuildPath = findFile(backendDistDir, 'build-info.json');

requireDir(frontendBuildDir);
requireDir(backendDistDir);
if (!backendDistBuildPath) {
  fail('Frozen backend is missing build-info.json; rebuild with scripts/build-all.bat');
}

const backendDistBuild = readJson(backendDistBuildPath);
const electronPkg = readJson(path.join(electronDir, 'package.json'));
const gitState = currentGitState();
const currentSourceFingerprint = sourceFingerprint();

if (gitState.sha && rootBuild.git_sha && rootBuild.git_sha !== gitState.sha) {
  fail(
    `root build-info was generated for ${rootBuild.git_short_sha || rootBuild.git_sha}, `
    + `but the current checkout is ${gitState.sha.slice(0, 12)}; run scripts\\build-all.bat first`,
  );
}
if (gitState.dirty !== null && typeof rootBuild.build_dirty === 'boolean' && rootBuild.build_dirty !== gitState.dirty) {
  fail(
    `root build-info dirty=${rootBuild.build_dirty} does not match the current checkout `
    + `dirty=${gitState.dirty}; run scripts\\build-all.bat first`,
  );
}
if (currentSourceFingerprint && rootBuild.source_fingerprint !== currentSourceFingerprint) {
  fail(
    `build-info source_fingerprint does not match the current checkout; `
    + `rebuild with scripts\\build-all.bat first`,
  );
}

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
