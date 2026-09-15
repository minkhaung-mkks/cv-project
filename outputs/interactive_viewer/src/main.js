import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import * as THREE from 'three';
import './style.css';

const viewerElement = document.querySelector('#viewer');
const loading = document.querySelector('#loading');
const loadingDetail = document.querySelector('#loading-detail');
const errorBox = document.querySelector('#error');
const modelName = document.querySelector('#model-name');
const modelSelect = document.querySelector('#model-select');
const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const playButton = document.querySelector('#play');
const playIcon = document.querySelector('#play-icon');
const stateDot = document.querySelector('#state-dot');
const stateLabel = document.querySelector('#state-label');

let viewer;
let objectUrl;
let autoRotate = true;
let loaded = false;
let homeDistance = 6;
const orbitCenter = new THREE.Vector3();
const pressed = new Set();

function createViewer() {
  return new GaussianSplats3D.Viewer({
    rootElement: viewerElement,
    cameraUp: [0, 1, 0],
    initialCameraPosition: [0, 0, 6],
    initialCameraLookAt: [0, 0, 0],
    sphericalHarmonicsDegree: 2,
    dynamicScene: false,
    sharedMemoryForWorkers: false,
    antialiased: true,
    selfDrivenMode: true,
    useBuiltInControls: true,
    sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant
  });
}

function cleanName(source) {
  const part = source.split('/').pop() || 'Model';
  return decodeURIComponent(part).replace(/\.(ply|splat|ksplat|spz)$/i, '').replaceAll('-', ' ');
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  window.setTimeout(() => { errorBox.hidden = true; }, 7000);
}

async function loadModel(source, label, format) {
  loaded = false;
  loading.classList.remove('hidden');
  loadingDetail.textContent = 'Preparing splats...';
  errorBox.hidden = true;

  try {
    if (viewer) {
      viewer.stop();
      viewer.dispose();
      viewerElement.replaceChildren();
    }
    viewer = createViewer();
    await viewer.addSplatScene(source, {
      format,
      splatAlphaRemovalThreshold: 3,
      showLoadingUI: false,
      progressiveLoad: true
    });
    viewer.start();
    viewer.controls.autoRotate = false;
    viewer.controls.enableDamping = false;
    viewer.controls.enablePan = false;
    frameModel();
    modelName.textContent = label || cleanName(source);
    loaded = true;
    loading.classList.add('hidden');
  } catch (error) {
    console.error(error);
    loading.classList.add('hidden');
    showError('This model could not be opened. Try a Gaussian-splat .ply or .splat file.');
  }
}

function frameModel() {
  const mesh = viewer?.getSplatMesh();
  if (!mesh || !viewer?.camera || !viewer?.controls) return;

  const bounds = new THREE.Box3();
  const point = new THREE.Vector3();
  const count = mesh.getSplatCount();
  for (let index = 0; index < count; index += 1) {
    mesh.getSplatCenter(index, point);
    bounds.expandByPoint(point);
  }

  if (bounds.isEmpty()) orbitCenter.set(0, 0, 0);
  else bounds.getCenter(orbitCenter);

  const radius = bounds.isEmpty() ? 2.5 : bounds.getSize(point).length() * 0.5;
  homeDistance = Math.max(0.5, radius * 2.4);
  resetView();
}

function orbit(horizontal, vertical) {
  if (!viewer?.camera || !viewer?.controls) return;
  const camera = viewer.camera;
  const target = orbitCenter;
  const offset = camera.position.clone().sub(target);
  const rotation = new THREE.Quaternion();

  if (horizontal) {
    rotation.setFromAxisAngle(camera.up.clone().normalize(), horizontal);
    offset.applyQuaternion(rotation);
  }

  if (vertical) {
    const viewDirection = offset.clone().negate().normalize();
    const screenRight = viewDirection.cross(camera.up).normalize();
    rotation.setFromAxisAngle(screenRight, vertical);
    offset.applyQuaternion(rotation);
    camera.up.applyQuaternion(rotation).normalize();
  }

  camera.position.copy(target).add(offset);
  viewer.controls.target.copy(target);
  camera.lookAt(target);
  viewer.controls.update();
}

function roll(angle) {
  if (!viewer?.camera || !viewer?.controls) return;
  const viewDirection = orbitCenter.clone().sub(viewer.camera.position).normalize();
  viewer.camera.up.applyAxisAngle(viewDirection, angle).normalize();
  viewer.controls.target.copy(orbitCenter);
  viewer.camera.lookAt(orbitCenter);
  viewer.controls.update();
}

function zoom(multiplier) {
  if (!viewer?.camera || !viewer?.controls) return;
  const target = orbitCenter;
  const offset = viewer.camera.position.clone().sub(target).multiplyScalar(multiplier);
  const distance = Math.max(homeDistance * 0.08, Math.min(homeDistance * 8, offset.length()));
  offset.setLength(distance);
  viewer.camera.position.copy(target).add(offset);
  viewer.controls.target.copy(target);
  viewer.camera.lookAt(target);
  viewer.controls.update();
}

function resetView() {
  if (!viewer?.camera || !viewer?.controls) return;
  viewer.controls.target.copy(orbitCenter);
  viewer.camera.position.copy(orbitCenter).add(new THREE.Vector3(0, 0, homeDistance));
  viewer.camera.up.set(0, 1, 0);
  viewer.camera.lookAt(orbitCenter);
  viewer.controls.update();
}

function updatePlayUI() {
  playIcon.textContent = autoRotate ? 'Ⅱ' : '▶';
  playButton.setAttribute('aria-label', autoRotate ? 'Pause auto-rotation' : 'Start auto-rotation');
  stateLabel.textContent = autoRotate ? 'Auto-rotating' : 'Paused';
  stateDot.classList.toggle('paused', !autoRotate);
  stateLabel.nextElementSibling.textContent = autoRotate ? 'Space to pause' : 'Space to play';
}

function togglePlay() {
  autoRotate = !autoRotate;
  updatePlayUI();
}

function animate() {
  if (loaded) {
    if (autoRotate && !pressed.size) orbit(0.0025, 0);
    const step = 0.018;
    if (pressed.has('KeyA') || pressed.has('ArrowLeft')) orbit(-step, 0);
    if (pressed.has('KeyD') || pressed.has('ArrowRight')) orbit(step, 0);
    if (pressed.has('KeyW') || pressed.has('ArrowUp')) orbit(0, -step);
    if (pressed.has('KeyS') || pressed.has('ArrowDown')) orbit(0, step);
    if (pressed.has('KeyQ')) roll(-step);
    if (pressed.has('KeyE')) roll(step);
    if (pressed.has('KeyZ')) zoom(1.018);
    if (pressed.has('KeyX')) zoom(0.982);
  }
  requestAnimationFrame(animate);
}

window.addEventListener('keydown', (event) => {
  if (event.target.matches('select, input, button')) return;
  if (['Space', 'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyZ', 'KeyX', 'KeyR', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.code)) event.preventDefault();
  if (event.code === 'Space' && !event.repeat) togglePlay();
  else if (event.code === 'KeyR' && !event.repeat) resetView();
  else pressed.add(event.code);
});
window.addEventListener('keyup', (event) => pressed.delete(event.code));
window.addEventListener('blur', () => pressed.clear());

playButton.addEventListener('click', togglePlay);
document.querySelector('#reset').addEventListener('click', resetView);
document.querySelector('#open-file').addEventListener('click', () => fileInput.click());
document.querySelector('#hide-help').addEventListener('click', () => document.querySelector('.controls').remove());
document.querySelector('#fullscreen').addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.querySelector('#stage').requestFullscreen();
});

modelSelect.addEventListener('change', () => loadModel(modelSelect.value, modelSelect.selectedOptions[0].text));
fileInput.addEventListener('change', () => openLocalFile(fileInput.files[0]));

function openLocalFile(file) {
  if (!file) return;
  if (!/\.(ply|splat|ksplat|spz)$/i.test(file.name)) {
    showError('Choose a .ply, .splat, .ksplat, or .spz model.');
    return;
  }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  modelSelect.value = '';
  const extension = file.name.split('.').pop().toLowerCase();
  const formats = {
    ply: GaussianSplats3D.SceneFormat.Ply,
    splat: GaussianSplats3D.SceneFormat.Splat,
    ksplat: GaussianSplats3D.SceneFormat.KSplat,
    spz: GaussianSplats3D.SceneFormat.Spz
  };
  loadModel(objectUrl, file.name.replace(/\.[^.]+$/, ''), formats[extension]);
}

for (const eventName of ['dragenter', 'dragover']) {
  window.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('active'); });
}
for (const eventName of ['dragleave', 'drop']) {
  window.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('active'); });
}
window.addEventListener('drop', (event) => openLocalFile(event.dataTransfer.files[0]));

updatePlayUI();
loadModel(modelSelect.value, modelSelect.selectedOptions[0].text);
animate();
