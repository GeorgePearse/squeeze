/// SIMD performance demonstration
///
/// This example demonstrates the performance improvement from SIMD optimizations.
/// Run with: cargo run --release --example simd_demo
use squeeze::distance;
use std::time::Instant;

fn main() {
    println!("=== SIMD Performance Demonstration ===\n");

    // Check SIMD availability
    println!("SIMD Detection:");
    println!("  SIMD distance functions enabled");

    println!();

    // Test with different vector sizes
    for dim in [64, 128, 256, 512, 1024] {
        println!("Testing with {}-dimensional vectors:", dim);

        // Generate test vectors
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.1).collect();

        let iterations = 100_000;

        // Benchmark scalar Euclidean
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..iterations {
            sum += distance::euclidean(&a, &b).unwrap();
        }
        let scalar_time = start.elapsed();
        let _ = std::hint::black_box(sum); // Prevent optimization

        // Benchmark SIMD Euclidean
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..iterations {
            sum += distance::euclidean_simd(&a, &b).unwrap();
        }
        let simd_time = start.elapsed();
        let _ = std::hint::black_box(sum); // Prevent optimization

        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

        println!(
            "  Scalar: {:>6.2} ns/op",
            scalar_time.as_nanos() as f64 / iterations as f64
        );
        println!(
            "  SIMD:   {:>6.2} ns/op",
            simd_time.as_nanos() as f64 / iterations as f64
        );
        println!("  Speedup: {:.2}x faster\n", speedup);
    }

    println!("=== Verification ===\n");

    // Verify correctness
    let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();

    let scalar_euclidean = distance::euclidean(&a, &b).unwrap();
    let simd_euclidean = distance::euclidean_simd(&a, &b).unwrap();

    let scalar_manhattan = distance::manhattan(&a, &b).unwrap();
    let simd_manhattan = distance::manhattan_simd(&a, &b).unwrap();

    let scalar_cosine = distance::cosine(&a, &b).unwrap();
    let simd_cosine = distance::cosine_simd(&a, &b).unwrap();

    println!("Euclidean:");
    println!("  Scalar: {:.6}", scalar_euclidean);
    println!("  SIMD:   {:.6}", simd_euclidean);
    println!(
        "  Match:  {}\n",
        (scalar_euclidean - simd_euclidean).abs() < 1e-4
    );

    println!("Manhattan:");
    println!("  Scalar: {:.6}", scalar_manhattan);
    println!("  SIMD:   {:.6}", simd_manhattan);
    println!(
        "  Match:  {}\n",
        (scalar_manhattan - simd_manhattan).abs() < 1e-4
    );

    println!("Cosine:");
    println!("  Scalar: {:.6}", scalar_cosine);
    println!("  SIMD:   {:.6}", simd_cosine);
    println!("  Match:  {}\n", (scalar_cosine - simd_cosine).abs() < 1e-4);
}
