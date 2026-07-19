use ndarray::array;
use squeeze::{algorithms::PCA, distance, neighbors::Hnsw};

#[test]
fn pca_is_available_through_the_public_algorithm_api() {
    let samples = array![
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 3.0],
        [3.0, 1.0, 5.0],
        [4.0, 2.0, 7.0],
    ];

    let embedding = PCA::new(2).fit_transform(&samples).unwrap();

    assert_eq!(embedding.dim(), (4, 2));
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn shared_primitives_are_available_through_public_modules() {
    let distance = distance::euclidean(&[0.0, 0.0], &[3.0, 4.0]).unwrap();
    let index = Hnsw::new(8, 32, 10, 42);

    assert_eq!(distance, 5.0);
    assert_eq!(index.m, 8);
}
