use std::cmp::Ordering;

/// Euclidean distance between two sparse vectors
pub fn sparse_euclidean(
    a_indices: &[i32],
    a_data: &[f32],
    b_indices: &[i32],
    b_data: &[f32],
) -> f32 {
    let mut sum_sq = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        let idx_a = a_indices[i];
        let idx_b = b_indices[j];

        match idx_a.cmp(&idx_b) {
            Ordering::Equal => {
                let diff = a_data[i] - b_data[j];
                sum_sq += diff * diff;
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                sum_sq += a_data[i] * a_data[i];
                i += 1;
            }
            Ordering::Greater => {
                sum_sq += b_data[j] * b_data[j];
                j += 1;
            }
        }
    }

    // Handle remaining elements
    while i < a_indices.len() {
        sum_sq += a_data[i] * a_data[i];
        i += 1;
    }
    while j < b_indices.len() {
        sum_sq += b_data[j] * b_data[j];
        j += 1;
    }

    sum_sq.sqrt()
}

/// Manhattan distance between two sparse vectors
pub fn sparse_manhattan(
    a_indices: &[i32],
    a_data: &[f32],
    b_indices: &[i32],
    b_data: &[f32],
) -> f32 {
    let mut sum_abs = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        let idx_a = a_indices[i];
        let idx_b = b_indices[j];

        match idx_a.cmp(&idx_b) {
            Ordering::Equal => {
                sum_abs += (a_data[i] - b_data[j]).abs();
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                sum_abs += a_data[i].abs();
                i += 1;
            }
            Ordering::Greater => {
                sum_abs += b_data[j].abs();
                j += 1;
            }
        }
    }

    while i < a_indices.len() {
        sum_abs += a_data[i].abs();
        i += 1;
    }
    while j < b_indices.len() {
        sum_abs += b_data[j].abs();
        j += 1;
    }

    sum_abs
}

/// Cosine distance between two sparse vectors
pub fn sparse_cosine(
    a_indices: &[i32],
    a_data: &[f32],
    b_indices: &[i32],
    b_data: &[f32],
) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_sq_a = 0.0;
    let mut norm_sq_b = 0.0;

    // Compute norm_a
    for &val in a_data {
        norm_sq_a += val * val;
    }

    // Compute norm_b
    for &val in b_data {
        norm_sq_b += val * val;
    }

    // Compute dot product (intersection)
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        let idx_a = a_indices[i];
        let idx_b = b_indices[j];

        match idx_a.cmp(&idx_b) {
            Ordering::Equal => {
                dot_product += a_data[i] * b_data[j];
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                i += 1;
            }
            Ordering::Greater => {
                j += 1;
            }
        }
    }

    let denom = (norm_sq_a * norm_sq_b).sqrt();

    if denom < 1e-10 {
        1.0
    } else {
        1.0 - (dot_product / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ============== sparse_euclidean tests ==============

    #[test]
    fn test_sparse_euclidean_identical_vectors() {
        let idx = vec![0, 2, 5];
        let data = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(sparse_euclidean(&idx, &data, &idx, &data), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_euclidean_disjoint_vectors() {
        // Vectors with no overlapping indices
        let a_idx = vec![0, 2];
        let a_data = vec![3.0, 4.0]; // ||a||^2 = 9 + 16 = 25
        let b_idx = vec![1, 3];
        let b_data = vec![5.0, 12.0]; // ||b||^2 = 25 + 144 = 169
        // Distance = sqrt(3^2 + 4^2 + 5^2 + 12^2) = sqrt(9+16+25+144) = sqrt(194)
        assert_relative_eq!(
            sparse_euclidean(&a_idx, &a_data, &b_idx, &b_data),
            194.0_f32.sqrt(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_sparse_euclidean_overlapping_vectors() {
        let a_idx = vec![0, 1, 2];
        let a_data = vec![1.0, 2.0, 3.0];
        let b_idx = vec![1, 2, 3];
        let b_data = vec![2.0, 1.0, 4.0];
        // idx=0: a=1, b=0 -> (1-0)^2 = 1
        // idx=1: a=2, b=2 -> (2-2)^2 = 0
        // idx=2: a=3, b=1 -> (3-1)^2 = 4
        // idx=3: a=0, b=4 -> (0-4)^2 = 16
        // sqrt(1+0+4+16) = sqrt(21)
        assert_relative_eq!(
            sparse_euclidean(&a_idx, &a_data, &b_idx, &b_data),
            21.0_f32.sqrt(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_sparse_euclidean_empty_vectors() {
        let empty_idx: Vec<i32> = vec![];
        let empty_data: Vec<f32> = vec![];
        assert_relative_eq!(
            sparse_euclidean(&empty_idx, &empty_data, &empty_idx, &empty_data),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_sparse_euclidean_one_empty() {
        let a_idx = vec![0, 1];
        let a_data = vec![3.0, 4.0]; // ||a|| = 5
        let empty_idx: Vec<i32> = vec![];
        let empty_data: Vec<f32> = vec![];
        // Distance to zero vector is the norm
        assert_relative_eq!(
            sparse_euclidean(&a_idx, &a_data, &empty_idx, &empty_data),
            5.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_sparse_euclidean_symmetric() {
        let a_idx = vec![0, 2, 4];
        let a_data = vec![1.0, 2.0, 3.0];
        let b_idx = vec![1, 2, 3];
        let b_data = vec![4.0, 5.0, 6.0];

        let d_ab = sparse_euclidean(&a_idx, &a_data, &b_idx, &b_data);
        let d_ba = sparse_euclidean(&b_idx, &b_data, &a_idx, &a_data);

        assert_relative_eq!(d_ab, d_ba, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_euclidean_non_negative() {
        let a_idx = vec![0, 1];
        let a_data = vec![-3.0, 4.0];
        let b_idx = vec![0, 2];
        let b_data = vec![5.0, -2.0];

        let dist = sparse_euclidean(&a_idx, &a_data, &b_idx, &b_data);
        assert!(dist >= 0.0, "Distance should be non-negative");
    }

    // ============== sparse_manhattan tests ==============

    #[test]
    fn test_sparse_manhattan_identical() {
        let idx = vec![0, 2, 5];
        let data = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(sparse_manhattan(&idx, &data, &idx, &data), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_manhattan_known_value() {
        let a_idx = vec![0, 1];
        let a_data = vec![1.0, 2.0];
        let b_idx = vec![0, 1];
        let b_data = vec![4.0, 6.0];
        // |1-4| + |2-6| = 3 + 4 = 7
        assert_relative_eq!(sparse_manhattan(&a_idx, &a_data, &b_idx, &b_data), 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_manhattan_disjoint() {
        let a_idx = vec![0];
        let a_data = vec![3.0];
        let b_idx = vec![1];
        let b_data = vec![4.0];
        // |3-0| + |0-4| = 3 + 4 = 7
        assert_relative_eq!(sparse_manhattan(&a_idx, &a_data, &b_idx, &b_data), 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_manhattan_symmetric() {
        let a_idx = vec![0, 2];
        let a_data = vec![1.0, 3.0];
        let b_idx = vec![1, 2];
        let b_data = vec![2.0, 5.0];

        let d_ab = sparse_manhattan(&a_idx, &a_data, &b_idx, &b_data);
        let d_ba = sparse_manhattan(&b_idx, &b_data, &a_idx, &a_data);

        assert_relative_eq!(d_ab, d_ba, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_manhattan_empty() {
        let empty_idx: Vec<i32> = vec![];
        let empty_data: Vec<f32> = vec![];
        assert_relative_eq!(
            sparse_manhattan(&empty_idx, &empty_data, &empty_idx, &empty_data),
            0.0,
            epsilon = 1e-6
        );
    }

    // ============== sparse_cosine tests ==============

    #[test]
    fn test_sparse_cosine_identical() {
        let idx = vec![0, 1, 2];
        let data = vec![1.0, 2.0, 3.0];
        // Identical vectors have cosine similarity = 1, distance = 0
        assert_relative_eq!(sparse_cosine(&idx, &data, &idx, &data), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_cosine_orthogonal() {
        // Orthogonal sparse vectors (no common indices)
        let a_idx = vec![0];
        let a_data = vec![1.0];
        let b_idx = vec![1];
        let b_data = vec![1.0];
        // Cosine similarity = 0, distance = 1
        assert_relative_eq!(sparse_cosine(&a_idx, &a_data, &b_idx, &b_data), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_cosine_parallel() {
        let idx = vec![0, 1, 2];
        let a_data = vec![1.0, 2.0, 3.0];
        let b_data = vec![2.0, 4.0, 6.0]; // 2x of a
        // Parallel vectors have cosine similarity = 1, distance = 0
        assert_relative_eq!(sparse_cosine(&idx, &a_data, &idx, &b_data), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_cosine_antiparallel() {
        let idx = vec![0, 1, 2];
        let a_data = vec![1.0, 2.0, 3.0];
        let b_data = vec![-1.0, -2.0, -3.0]; // -1x of a
        // Anti-parallel vectors have cosine similarity = -1, distance = 2
        assert_relative_eq!(sparse_cosine(&idx, &a_data, &idx, &b_data), 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_cosine_zero_vector() {
        let a_idx = vec![0];
        let a_data = vec![1.0];
        let zero_idx: Vec<i32> = vec![];
        let zero_data: Vec<f32> = vec![];
        // Distance to zero vector should be 1.0 (undefined, defaulting to max distance)
        assert_relative_eq!(sparse_cosine(&a_idx, &a_data, &zero_idx, &zero_data), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sparse_cosine_symmetric() {
        let a_idx = vec![0, 2];
        let a_data = vec![1.0, 3.0];
        let b_idx = vec![1, 2];
        let b_data = vec![2.0, 5.0];

        let d_ab = sparse_cosine(&a_idx, &a_data, &b_idx, &b_data);
        let d_ba = sparse_cosine(&b_idx, &b_data, &a_idx, &a_data);

        assert_relative_eq!(d_ab, d_ba, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_cosine_bounded() {
        // Cosine distance should be in [0, 2]
        let a_idx = vec![0, 1, 2];
        let a_data = vec![1.0, -2.0, 3.0];
        let b_idx = vec![0, 2, 3];
        let b_data = vec![-1.0, 2.0, -1.0];

        let dist = sparse_cosine(&a_idx, &a_data, &b_idx, &b_data);
        assert!(dist >= 0.0 && dist <= 2.0, "Cosine distance {} out of bounds [0, 2]", dist);
    }

    #[test]
    fn test_sparse_cosine_known_value() {
        // [1, 0] and [1, 1] have cosine = 1/sqrt(2) ≈ 0.707
        // cosine distance = 1 - 0.707 ≈ 0.293
        let a_idx = vec![0];
        let a_data = vec![1.0];
        let b_idx = vec![0, 1];
        let b_data = vec![1.0, 1.0];

        let dist = sparse_cosine(&a_idx, &a_data, &b_idx, &b_data);
        let expected = 1.0 - 1.0 / (2.0_f32).sqrt();
        assert_relative_eq!(dist, expected, epsilon = 1e-5);
    }
}
