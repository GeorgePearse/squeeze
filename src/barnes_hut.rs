//! Barnes-Hut tree for O(n log n) t-SNE gradient computation
//!
//! The Barnes-Hut algorithm approximates long-range forces by treating
//! distant groups of points as single points located at their center of mass.

use ndarray::{Array1, Array2};

/// Axis-aligned bounding box for spatial partitioning
#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        Self { min_x, max_x, min_y, max_y }
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Get the quadrant (0-3) for a point relative to center
    pub fn get_quadrant(&self, x: f64, y: f64) -> usize {
        let cx = (self.min_x + self.max_x) / 2.0;
        let cy = (self.min_y + self.max_y) / 2.0;
        
        let mut quad = 0;
        if x > cx { quad += 1; }
        if y > cy { quad += 2; }
        quad
    }

    /// Get bounding box for a specific quadrant
    pub fn get_quadrant_bounds(&self, quadrant: usize) -> BoundingBox {
        let cx = (self.min_x + self.max_x) / 2.0;
        let cy = (self.min_y + self.max_y) / 2.0;
        
        match quadrant {
            0 => BoundingBox::new(self.min_x, cx, self.min_y, cy),
            1 => BoundingBox::new(cx, self.max_x, self.min_y, cy),
            2 => BoundingBox::new(self.min_x, cx, cy, self.max_y),
            3 => BoundingBox::new(cx, self.max_x, cy, self.max_y),
            _ => panic!("Invalid quadrant"),
        }
    }
}

/// Barnes-Hut quadtree node for 2D embeddings
pub struct QuadTreeNode {
    /// Center of mass for all points in this node
    pub center_of_mass: [f64; 2],
    /// Total mass (number of points) in this node
    pub total_mass: f64,
    /// Bounding box for this node
    pub bounds: BoundingBox,
    /// Child nodes (None for leaf, Some for internal)
    pub children: Option<Box<[Option<QuadTreeNode>; 4]>>,
    /// Point index if this is a leaf with a single point
    pub point_idx: Option<usize>,
}

impl QuadTreeNode {
    /// Build a quadtree from a set of 2D points
    pub fn build(points: &Array2<f64>) -> Self {
        let n = points.nrows();
        if n == 0 {
            panic!("Cannot build tree from empty points");
        }

        // Find bounding box
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for i in 0..n {
            let x = points[[i, 0]];
            let y = points[[i, 1]];
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        // Add small margin to avoid points on boundaries
        let margin = 1e-5;
        let bounds = BoundingBox::new(
            min_x - margin,
            max_x + margin,
            min_y - margin,
            max_y + margin,
        );

        // Build tree recursively
        let mut root = QuadTreeNode {
            center_of_mass: [0.0, 0.0],
            total_mass: 0.0,
            bounds,
            children: None,
            point_idx: None,
        };

        for i in 0..n {
            root.insert(i, points[[i, 0]], points[[i, 1]]);
        }

        root
    }

    /// Insert a point into the tree
    fn insert(&mut self, idx: usize, x: f64, y: f64) {
        // Update center of mass incrementally
        let new_mass = self.total_mass + 1.0;
        self.center_of_mass[0] = (self.center_of_mass[0] * self.total_mass + x) / new_mass;
        self.center_of_mass[1] = (self.center_of_mass[1] * self.total_mass + y) / new_mass;
        self.total_mass = new_mass;

        // If this is an empty leaf, just store the point
        if self.total_mass == 1.0 {
            self.point_idx = Some(idx);
            return;
        }

        // If this is a leaf with one point, need to split
        if self.children.is_none() {
            self.subdivide();
            
            // Reinsert the existing point
            if let Some(old_idx) = self.point_idx.take() {
                // Use the center of mass from before this insertion as the old point location
                let old_x = self.center_of_mass[0] * self.total_mass - x;
                let old_y = self.center_of_mass[1] * self.total_mass - y;
                let old_quad = self.bounds.get_quadrant(old_x, old_y);
                if let Some(ref mut children) = self.children {
                    if children[old_quad].is_none() {
                        children[old_quad] = Some(QuadTreeNode {
                            center_of_mass: [old_x, old_y],
                            total_mass: 1.0,
                            bounds: self.bounds.get_quadrant_bounds(old_quad),
                            children: None,
                            point_idx: Some(old_idx),
                        });
                    } else {
                        children[old_quad].as_mut().unwrap().insert(old_idx, old_x, old_y);
                    }
                }
            }
        }

        // Insert new point into appropriate quadrant
        let quad = self.bounds.get_quadrant(x, y);
        if let Some(ref mut children) = self.children {
            if children[quad].is_none() {
                children[quad] = Some(QuadTreeNode {
                    center_of_mass: [x, y],
                    total_mass: 1.0,
                    bounds: self.bounds.get_quadrant_bounds(quad),
                    children: None,
                    point_idx: Some(idx),
                });
            } else {
                children[quad].as_mut().unwrap().insert(idx, x, y);
            }
        }
    }

    /// Subdivide this node into 4 quadrants
    fn subdivide(&mut self) {
        let mut children: [Option<QuadTreeNode>; 4] = [None, None, None, None];
        self.children = Some(Box::new(children));
    }

    /// Compute repulsive forces using Barnes-Hut approximation
    pub fn compute_non_edge_forces(
        &self,
        point: &[f64],
        theta: f64,
        point_idx: usize,
    ) -> [f64; 2] {
        // Skip if this is the same point
        if let Some(idx) = self.point_idx {
            if idx == point_idx {
                return [0.0, 0.0];
            }
        }

        let dx = self.center_of_mass[0] - point[0];
        let dy = self.center_of_mass[1] - point[1];
        let dist_sq = dx * dx + dy * dy;

        // If node is far enough away, use approximation
        let node_size = self.bounds.width().max(self.bounds.height());
        if node_size * node_size / dist_sq < theta * theta {
            // Compute force from center of mass
            let inv_dist = 1.0 / (1.0 + dist_sq);
            let force_scalar = self.total_mass * inv_dist * inv_dist;
            return [force_scalar * dx, force_scalar * dy];
        }

        // Otherwise, recurse into children
        if let Some(ref children) = self.children {
            let mut force = [0.0, 0.0];
            for child in children.iter() {
                if let Some(ref child_node) = child {
                    let child_force = child_node.compute_non_edge_forces(point, theta, point_idx);
                    force[0] += child_force[0];
                    force[1] += child_force[1];
                }
            }
            return force;
        }

        // Leaf node with single point - compute exact force
        if self.point_idx.is_some() && self.point_idx.unwrap() != point_idx {
            let inv_dist = 1.0 / (1.0 + dist_sq);
            let force_scalar = inv_dist * inv_dist;
            return [force_scalar * dx, force_scalar * dy];
        }

        [0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 10.0);
        
        assert_eq!(bbox.width(), 10.0);
        assert_eq!(bbox.height(), 10.0);
        
        assert!(bbox.contains(5.0, 5.0));
        assert!(!bbox.contains(-1.0, 5.0));
        assert!(!bbox.contains(11.0, 5.0));
        
        // Test quadrant determination
        assert_eq!(bbox.get_quadrant(2.0, 2.0), 0);
        assert_eq!(bbox.get_quadrant(7.0, 2.0), 1);
        assert_eq!(bbox.get_quadrant(2.0, 7.0), 2);
        assert_eq!(bbox.get_quadrant(7.0, 7.0), 3);
    }

    #[test]
    fn test_quadtree_build() {
        let points = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]).unwrap();
        
        let tree = QuadTreeNode::build(&points);
        
        assert_eq!(tree.total_mass, 4.0);
        assert_relative_eq!(tree.center_of_mass[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(tree.center_of_mass[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_barnes_hut_forces() {
        let points = Array2::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            10.0, 0.0,
            5.0, 5.0,
        ]).unwrap();
        
        let tree = QuadTreeNode::build(&points);
        
        // Test force computation with high theta (more approximation)
        let force = tree.compute_non_edge_forces(&[0.0, 0.0], 0.5, 0);
        
        // Forces should be non-zero (repulsive from other points)
        assert!(force[0] != 0.0 || force[1] != 0.0);
    }
}