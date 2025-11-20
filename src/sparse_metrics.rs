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
