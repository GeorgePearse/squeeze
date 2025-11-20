use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::hint::black_box as hint_black_box;

// Import from the library (need to set up proper module structure)
// For now, we'll define benchmark helpers

/// Generate random f32 vectors
fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Euclidean distance (reference implementation)
fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Manhattan distance (reference implementation)
fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Cosine distance (reference implementation)
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}

/// Benchmark distance metric computation
fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");
    
    for dim in [8, 16, 32, 64, 128, 256, 512, 1024].iter() {
        let vectors = generate_vectors(2, *dim);
        let a = &vectors[0];
        let b = &vectors[1];
        
        group.throughput(Throughput::Elements(*dim as u64));
        
        group.bench_with_input(BenchmarkId::new("euclidean", dim), dim, |bench, _| {
            bench.iter(|| euclidean(black_box(a), black_box(b)));
        });
        
        group.bench_with_input(BenchmarkId::new("manhattan", dim), dim, |bench, _| {
            bench.iter(|| manhattan(black_box(a), black_box(b)));
        });
        
        group.bench_with_input(BenchmarkId::new("cosine", dim), dim, |bench, _| {
            bench.iter(|| cosine(black_box(a), black_box(b)));
        });
    }
    
    group.finish();
}

/// Benchmark batch distance computation
fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    
    for n in [10, 50, 100, 500, 1000].iter() {
        let vectors = generate_vectors(*n + 1, 128);
        let query = &vectors[0];
        let data = &vectors[1..];
        
        group.throughput(Throughput::Elements(*n as u64));
        
        group.bench_with_input(BenchmarkId::new("euclidean_batch", n), n, |bench, _| {
            bench.iter(|| {
                for vec in data {
                    hint_black_box(euclidean(black_box(query), black_box(vec)));
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark neighbor selection (heap operations)
fn bench_neighbor_selection(c: &mut Criterion) {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;
    
    #[derive(Clone, Copy, Debug)]
    #[allow(dead_code)] // Used for benchmarking only
    struct Entry {
        index: usize,
        distance: f32,
    }
    
    impl PartialEq for Entry {
        fn eq(&self, other: &Self) -> bool {
            self.distance.to_bits() == other.distance.to_bits()
        }
    }
    
    impl Eq for Entry {}
    
    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    
    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.distance.partial_cmp(&other.distance)
                .unwrap_or(Ordering::Equal)
                .reverse() // Max heap
        }
    }
    
    let mut group = c.benchmark_group("neighbor_selection");
    
    for k in [5, 10, 15, 30, 50].iter() {
        let mut rng = rand::thread_rng();
        let candidates: Vec<Entry> = (0..1000)
            .map(|i| Entry { index: i, distance: rng.random() })
            .collect();
        
        group.bench_with_input(BenchmarkId::new("heap_insert", k), k, |bench, k| {
            bench.iter(|| {
                let mut heap = BinaryHeap::with_capacity(*k);
                for &entry in candidates.iter().take(100) {
                    if heap.len() < *k {
                        heap.push(entry);
                    } else if let Some(worst) = heap.peek() {
                        if entry.distance < worst.distance {
                            heap.pop();
                            heap.push(entry);
                        }
                    }
                }
                hint_black_box(heap)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("partial_sort", k), k, |bench, k| {
            bench.iter(|| {
                let mut sorted = candidates[..100].to_vec();
                sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                sorted.truncate(*k);
                hint_black_box(sorted)
            });
        });
    }
    
    group.finish();
}

/// Benchmark HNSW construction (synthetic)
fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_construction");
    group.sample_size(10); // Reduce sample size for slow benchmarks
    
    for n in [100, 500, 1000].iter() {
        let vectors = generate_vectors(*n, 64);
        
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |bench, _| {
            bench.iter(|| {
                // Simulate construction by computing all-pairs distances
                let mut count = 0;
                for i in 0..*n {
                    for j in (i+1)..*n {
                        hint_black_box(euclidean(&vectors[i], &vectors[j]));
                        count += 1;
                    }
                }
                hint_black_box(count)
            });
        });
    }
    
    group.finish();
}

/// Benchmark HNSW search (synthetic)
fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    
    let n = 1000;
    let vectors = generate_vectors(n + 10, 64);
    let data = &vectors[..n];
    let queries = &vectors[n..];
    
    for k in [5, 10, 15, 30].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |bench, k| {
            bench.iter(|| {
                // Simulate search by finding k nearest neighbors
                for query in queries {
                    let mut distances: Vec<(usize, f32)> = data
                        .iter()
                        .enumerate()
                        .map(|(i, vec)| (i, euclidean(query, vec)))
                        .collect();
                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    distances.truncate(*k);
                    hint_black_box(distances);
                }
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_distance_metrics,
    bench_batch_distances,
    bench_neighbor_selection,
    bench_hnsw_construction,
    bench_hnsw_search,
);

criterion_main!(benches);
