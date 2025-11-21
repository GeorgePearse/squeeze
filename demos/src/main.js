import createScatterplot from 'regl-scatterplot';

// Color palettes for different class counts
const PALETTES = {
  10: [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
  ],
  20: [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
  ]
};

// Dataset configurations
const DATASETS = {
  digits: {
    name: 'MNIST Digits',
    file: 'digits',
    classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    description: 'Handwritten digits from sklearn'
  },
  fashion: {
    name: 'Fashion MNIST',
    file: 'fashion',
    classes: ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    description: 'Fashion product images'
  },
  cifar10: {
    name: 'CIFAR-10',
    file: 'cifar10',
    classes: ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck'],
    description: 'Tiny images in 10 classes'
  },
  newsgroups: {
    name: '20 Newsgroups',
    file: 'newsgroups',
    classes: Array.from({length: 20}, (_, i) => `Group ${i}`),
    description: 'Text document embeddings'
  }
};

// Global state
let scatterplot = null;
let currentData = null;
let currentDataset = 'digits';
let currentAlgorithm = 'umap';

// Initialize scatterplot
function initScatterplot() {
  const canvas = document.getElementById('scatterplot');
  const container = document.getElementById('canvas-container');
  
  // Set canvas size
  const resize = () => {
    canvas.width = container.clientWidth * window.devicePixelRatio;
    canvas.height = container.clientHeight * window.devicePixelRatio;
    canvas.style.width = `${container.clientWidth}px`;
    canvas.style.height = `${container.clientHeight}px`;
  };
  resize();
  window.addEventListener('resize', resize);

  scatterplot = createScatterplot({
    canvas,
    width: container.clientWidth,
    height: container.clientHeight,
    pointSize: 3,
    opacity: 0.8,
    lassoOnLongPress: true,
  });

  // Handle point hover
  scatterplot.subscribe('pointover', (pointIndex) => {
    if (currentData && pointIndex >= 0) {
      updateSelectedInfo(pointIndex);
    }
  });

  scatterplot.subscribe('pointout', () => {
    document.getElementById('selected-info').innerHTML = 
      '<p>Hover over a point to see details</p>';
  });

  return scatterplot;
}

// Load embedding data
async function loadData(dataset, algorithm) {
  const datasetConfig = DATASETS[dataset];
  const url = `/data/${datasetConfig.file}_${algorithm}.json`;
  
  showLoading(true);
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load ${url}`);
    }
    const data = await response.json();
    currentData = data;
    return data;
  } catch (error) {
    console.error('Error loading data:', error);
    showError(`Could not load ${dataset} with ${algorithm}. Run 'npm run generate' first.`);
    return null;
  } finally {
    showLoading(false);
  }
}

// Update the scatterplot with new data
function updateScatterplot(data) {
  if (!scatterplot || !data) return;

  const { points, labels, metadata } = data;
  const nClasses = Math.max(...labels) + 1;
  const palette = nClasses <= 10 ? PALETTES[10] : PALETTES[20];
  
  // Convert hex colors to [r, g, b, a] format
  const colorMap = palette.slice(0, nClasses).map(hex => {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return [r, g, b, 1];
  });

  // Assign colors based on labels
  const colors = labels.map(label => colorMap[label % colorMap.length]);

  scatterplot.draw({
    x: points.map(p => p[0]),
    y: points.map(p => p[1]),
    color: colors,
  });

  // Update info panel
  updateDatasetInfo(data);
  updateLegend(data);
}

// Update dataset info panel
function updateDatasetInfo(data) {
  const { labels, metadata } = data;
  const nClasses = new Set(labels).size;
  
  document.getElementById('n-points').textContent = labels.length.toLocaleString();
  document.getElementById('n-classes').textContent = nClasses;
  document.getElementById('current-algo').textContent = currentAlgorithm.toUpperCase();
  document.getElementById('compute-time').textContent = 
    metadata?.compute_time ? `${metadata.compute_time.toFixed(2)}s` : '-';
}

// Update legend
function updateLegend(data) {
  const { labels } = data;
  const datasetConfig = DATASETS[currentDataset];
  const nClasses = new Set(labels).size;
  const palette = nClasses <= 10 ? PALETTES[10] : PALETTES[20];
  
  const legendEl = document.getElementById('legend');
  legendEl.innerHTML = '';
  
  for (let i = 0; i < Math.min(nClasses, datasetConfig.classes.length); i++) {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `
      <span class="legend-color" style="background: ${palette[i]}"></span>
      <span class="legend-label">${datasetConfig.classes[i]}</span>
    `;
    legendEl.appendChild(item);
  }
}

// Update selected point info
function updateSelectedInfo(pointIndex) {
  if (!currentData) return;
  
  const { labels } = currentData;
  const datasetConfig = DATASETS[currentDataset];
  const label = labels[pointIndex];
  const className = datasetConfig.classes[label] || `Class ${label}`;
  
  document.getElementById('selected-info').innerHTML = `
    <p><strong>Index:</strong> ${pointIndex}</p>
    <p><strong>Class:</strong> ${className}</p>
    <p><strong>Label:</strong> ${label}</p>
  `;
}

// Show/hide loading state
function showLoading(show) {
  const container = document.getElementById('canvas-container');
  let loading = container.querySelector('.loading');
  
  if (show && !loading) {
    loading = document.createElement('div');
    loading.className = 'loading';
    loading.innerHTML = `
      <div class="loading-spinner"></div>
      <p>Loading embedding...</p>
    `;
    container.appendChild(loading);
  } else if (!show && loading) {
    loading.remove();
  }
}

// Show error message
function showError(message) {
  const container = document.getElementById('canvas-container');
  let error = container.querySelector('.error');
  
  if (!error) {
    error = document.createElement('div');
    error.className = 'loading';
    container.appendChild(error);
  }
  
  error.innerHTML = `<p style="color: #f85149;">${message}</p>`;
}

// Handle control changes
function setupControls() {
  const datasetSelect = document.getElementById('dataset-select');
  const algorithmSelect = document.getElementById('algorithm-select');
  const colorSelect = document.getElementById('color-select');
  const pointSizeSlider = document.getElementById('point-size');
  const opacitySlider = document.getElementById('opacity');

  datasetSelect.addEventListener('change', async (e) => {
    currentDataset = e.target.value;
    const data = await loadData(currentDataset, currentAlgorithm);
    if (data) updateScatterplot(data);
  });

  algorithmSelect.addEventListener('change', async (e) => {
    currentAlgorithm = e.target.value;
    const data = await loadData(currentDataset, currentAlgorithm);
    if (data) updateScatterplot(data);
  });

  pointSizeSlider.addEventListener('input', (e) => {
    const size = parseFloat(e.target.value);
    document.getElementById('point-size-value').textContent = size;
    if (scatterplot) {
      scatterplot.set({ pointSize: size });
    }
  });

  opacitySlider.addEventListener('input', (e) => {
    const opacity = parseFloat(e.target.value);
    document.getElementById('opacity-value').textContent = opacity;
    if (scatterplot) {
      scatterplot.set({ opacity });
    }
  });
}

// Main initialization
async function main() {
  initScatterplot();
  setupControls();
  
  // Load initial data
  const data = await loadData(currentDataset, currentAlgorithm);
  if (data) {
    updateScatterplot(data);
  }
}

// Start the app
main();
