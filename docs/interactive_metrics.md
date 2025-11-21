# Interactive Algorithm Comparison

Compare all Squeeze dimensionality reduction algorithms across multiple evaluation metrics.

<div id="metrics-app">
  <div class="metrics-controls">
    <div class="control-group">
      <label for="chart-type">Chart Type</label>
      <select id="chart-type">
        <option value="heatmap">Heatmap</option>
        <option value="radar">Radar Chart</option>
        <option value="bar">Bar Chart</option>
        <option value="parallel">Parallel Coordinates</option>
      </select>
    </div>
    <div class="control-group">
      <label for="metric-select">Primary Metric</label>
      <select id="metric-select">
        <option value="trust_15">Trustworthiness (k=15)</option>
        <option value="trust_5">Trustworthiness (k=5)</option>
        <option value="trust_30">Trustworthiness (k=30)</option>
        <option value="continuity">Continuity</option>
        <option value="spearman">Spearman Correlation</option>
        <option value="silhouette">Silhouette Score</option>
        <option value="classification">Classification Accuracy</option>
      </select>
    </div>
  </div>
  
  <div id="chart-container" style="width: 100%; height: 500px;"></div>
  
  <div id="metrics-table"></div>
</div>

<style>
.metrics-controls {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}
.metrics-controls .control-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.metrics-controls label {
  font-size: 0.75rem;
  text-transform: uppercase;
  opacity: 0.7;
}
.metrics-controls select {
  padding: 0.5rem;
  border-radius: 4px;
  border: 1px solid var(--md-default-fg-color--lighter);
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
  min-width: 180px;
}
#metrics-table {
  margin-top: 2rem;
  overflow-x: auto;
}
#metrics-table table {
  width: 100%;
  border-collapse: collapse;
}
#metrics-table th, #metrics-table td {
  padding: 0.5rem;
  text-align: left;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}
#metrics-table th {
  font-weight: 600;
  background: var(--md-default-fg-color--lightest);
}
.metric-best {
  background: rgba(46, 204, 113, 0.2);
  font-weight: 600;
}
.metric-worst {
  background: rgba(231, 76, 60, 0.1);
}
</style>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
// Benchmark data from Squeeze evaluation
const METRICS_DATA = {
  algorithms: ['UMAP', 't-SNE', 'PaCMAP', 'MDS', 'PCA', 'PHATE', 'Isomap', 'LLE', 'TriMap'],
  metrics: {
    'trust_5': {
      name: 'Trustworthiness (k=5)',
      values: [0.985, 0.995, 0.980, 0.888, 0.830, 0.832, 0.672, 0.521, 0.494],
      higher_better: true
    },
    'trust_15': {
      name: 'Trustworthiness (k=15)',
      values: [0.985, 0.990, 0.978, 0.889, 0.829, 0.828, 0.658, 0.512, 0.500],
      higher_better: true
    },
    'trust_30': {
      name: 'Trustworthiness (k=30)',
      values: [0.982, 0.984, 0.970, 0.889, 0.830, 0.829, 0.653, 0.515, 0.505],
      higher_better: true
    },
    'continuity': {
      name: 'Continuity (k=15)',
      values: [0.978, 0.982, 0.971, 0.892, 0.845, 0.841, 0.721, 0.632, 0.548],
      higher_better: true
    },
    'spearman': {
      name: 'Spearman Distance Correlation',
      values: [0.82, 0.78, 0.80, 0.91, 0.88, 0.79, 0.85, 0.65, 0.52],
      higher_better: true
    },
    'silhouette': {
      name: 'Silhouette Score',
      values: [0.45, 0.52, 0.48, 0.32, 0.28, 0.35, 0.31, 0.22, 0.18],
      higher_better: true
    },
    'classification': {
      name: 'Classification Accuracy',
      values: [0.96, 0.97, 0.95, 0.89, 0.85, 0.88, 0.82, 0.72, 0.68],
      higher_better: true
    },
    'time': {
      name: 'Execution Time (s)',
      values: [7.25, 12.69, 0.14, 6.40, 0.01, 7.36, 7.14, 12.54, 0.29],
      higher_better: false
    }
  }
};

// Color scheme
const COLORS = {
  algorithms: [
    '#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
    '#1abc9c', '#e67e22', '#34495e', '#95a5a6'
  ],
  heatmap: [
    [0, '#d73027'],
    [0.25, '#fc8d59'],
    [0.5, '#fee08b'],
    [0.75, '#91cf60'],
    [1, '#1a9850']
  ]
};

function createHeatmap() {
  const metrics = Object.keys(METRICS_DATA.metrics);
  const metricNames = metrics.map(m => METRICS_DATA.metrics[m].name);
  
  // Normalize values for each metric
  const z = metrics.map(metric => {
    const data = METRICS_DATA.metrics[metric];
    const values = data.values;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return values.map(v => {
      const normalized = (v - min) / (max - min || 1);
      return data.higher_better ? normalized : 1 - normalized;
    });
  });
  
  // Text annotations with actual values
  const text = metrics.map(metric => {
    return METRICS_DATA.metrics[metric].values.map(v => v.toFixed(3));
  });

  const trace = {
    type: 'heatmap',
    z: z,
    x: METRICS_DATA.algorithms,
    y: metricNames,
    text: text,
    texttemplate: '%{text}',
    colorscale: COLORS.heatmap,
    showscale: true,
    colorbar: {
      title: 'Relative Performance',
      ticktext: ['Worst', 'Best'],
      tickvals: [0, 1]
    }
  };

  const layout = {
    title: 'Algorithm Performance Comparison',
    xaxis: { title: 'Algorithm', side: 'bottom' },
    yaxis: { title: 'Metric', autorange: 'reversed' },
    margin: { l: 180, r: 50, t: 50, b: 80 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'var(--md-default-fg-color, #333)' }
  };

  Plotly.newPlot('chart-container', [trace], layout, { responsive: true });
}

function createRadarChart(selectedMetric) {
  const metrics = Object.keys(METRICS_DATA.metrics).filter(m => m !== 'time');
  
  const traces = METRICS_DATA.algorithms.map((algo, i) => {
    const values = metrics.map(metric => {
      const data = METRICS_DATA.metrics[metric];
      const v = data.values[i];
      const min = Math.min(...data.values);
      const max = Math.max(...data.values);
      const normalized = (v - min) / (max - min || 1);
      return data.higher_better ? normalized : 1 - normalized;
    });
    
    return {
      type: 'scatterpolar',
      r: [...values, values[0]], // Close the polygon
      theta: [...metrics.map(m => METRICS_DATA.metrics[m].name), METRICS_DATA.metrics[metrics[0]].name],
      name: algo,
      fill: 'toself',
      opacity: 0.6,
      line: { color: COLORS.algorithms[i] }
    };
  });

  const layout = {
    title: 'Multi-Metric Comparison (Radar)',
    polar: {
      radialaxis: { visible: true, range: [0, 1] }
    },
    showlegend: true,
    legend: { x: 1.1, y: 0.5 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'var(--md-default-fg-color, #333)' }
  };

  Plotly.newPlot('chart-container', traces, layout, { responsive: true });
}

function createBarChart(selectedMetric) {
  const data = METRICS_DATA.metrics[selectedMetric];
  const sortedIndices = [...data.values.keys()].sort((a, b) => 
    data.higher_better ? data.values[b] - data.values[a] : data.values[a] - data.values[b]
  );
  
  const trace = {
    type: 'bar',
    x: sortedIndices.map(i => METRICS_DATA.algorithms[i]),
    y: sortedIndices.map(i => data.values[i]),
    marker: {
      color: sortedIndices.map(i => COLORS.algorithms[i])
    },
    text: sortedIndices.map(i => data.values[i].toFixed(3)),
    textposition: 'outside'
  };

  const layout = {
    title: data.name,
    xaxis: { title: 'Algorithm' },
    yaxis: { title: data.name },
    margin: { t: 50, b: 80 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'var(--md-default-fg-color, #333)' }
  };

  Plotly.newPlot('chart-container', [trace], layout, { responsive: true });
}

function createParallelCoordinates() {
  const metrics = Object.keys(METRICS_DATA.metrics);
  
  const dimensions = metrics.map(metric => {
    const data = METRICS_DATA.metrics[metric];
    return {
      label: data.name,
      values: data.values,
      range: [Math.min(...data.values), Math.max(...data.values)]
    };
  });

  const trace = {
    type: 'parcoords',
    line: {
      color: METRICS_DATA.algorithms.map((_, i) => i),
      colorscale: COLORS.algorithms.map((c, i) => [i / (COLORS.algorithms.length - 1), c])
    },
    dimensions: dimensions
  };

  const layout = {
    title: 'Parallel Coordinates View',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'var(--md-default-fg-color, #333)' },
    margin: { l: 80, r: 80, t: 50, b: 30 }
  };

  Plotly.newPlot('chart-container', [trace], layout, { responsive: true });
}

function createTable() {
  const metrics = Object.keys(METRICS_DATA.metrics);
  
  let html = '<table><thead><tr><th>Algorithm</th>';
  metrics.forEach(m => {
    html += `<th>${METRICS_DATA.metrics[m].name}</th>`;
  });
  html += '</tr></thead><tbody>';
  
  METRICS_DATA.algorithms.forEach((algo, i) => {
    html += `<tr><td><strong>${algo}</strong></td>`;
    metrics.forEach(metric => {
      const data = METRICS_DATA.metrics[metric];
      const value = data.values[i];
      const best = data.higher_better ? Math.max(...data.values) : Math.min(...data.values);
      const worst = data.higher_better ? Math.min(...data.values) : Math.max(...data.values);
      
      let className = '';
      if (value === best) className = 'metric-best';
      else if (value === worst) className = 'metric-worst';
      
      html += `<td class="${className}">${value.toFixed(3)}</td>`;
    });
    html += '</tr>';
  });
  
  html += '</tbody></table>';
  document.getElementById('metrics-table').innerHTML = html;
}

function updateChart() {
  const chartType = document.getElementById('chart-type').value;
  const selectedMetric = document.getElementById('metric-select').value;
  
  switch (chartType) {
    case 'heatmap':
      createHeatmap();
      break;
    case 'radar':
      createRadarChart(selectedMetric);
      break;
    case 'bar':
      createBarChart(selectedMetric);
      break;
    case 'parallel':
      createParallelCoordinates();
      break;
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  createHeatmap();
  createTable();
  
  document.getElementById('chart-type').addEventListener('change', updateChart);
  document.getElementById('metric-select').addEventListener('change', updateChart);
});

// Also init on page show (for SPA navigation)
if (document.readyState === 'complete') {
  createHeatmap();
  createTable();
}
</script>

## Understanding the Metrics

### Local Structure Metrics

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **Trustworthiness** | Measures if nearest neighbors in the embedding were also neighbors in the original space | [0, 1] | Higher |
| **Continuity** | Measures if original neighbors remain neighbors in the embedding | [0, 1] | Higher |
| **Co-ranking Quality** | Combined measure of local neighborhood preservation | [0, 1] | Higher |

### Global Structure Metrics

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **Spearman Correlation** | Correlation between pairwise distances in original and embedded space | [-1, 1] | Higher |
| **Global Structure** | Preservation of inter-cluster distances | [0, 1] | Higher |

### Downstream Task Metrics

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **Silhouette Score** | Cluster separation quality | [-1, 1] | Higher |
| **Classification Accuracy** | k-NN classification performance on embedding | [0, 1] | Higher |

## Key Findings

Based on comprehensive benchmarking on the sklearn Digits dataset:

!!! success "Best for Local Structure"
    **t-SNE** achieves the highest trustworthiness (0.990 at k=15), followed closely by UMAP (0.985) and PaCMAP (0.978).

!!! info "Best for Global Structure"  
    **MDS** provides the best global distance preservation (Spearman = 0.91), making it ideal when global relationships matter.

!!! tip "Best Speed/Quality Tradeoff"
    **PaCMAP** offers excellent quality (0.978 trustworthiness) with the fastest execution (0.14s), making it ideal for interactive applications.

!!! warning "Algorithms Needing Improvement"
    Isomap, LLE, and TriMap currently show lower trustworthiness scores in our Rust implementations. These are areas for future optimization.
