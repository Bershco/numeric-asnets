;;Instance with 3x2x3 points
(define (problem droneprob_3_2_3_5) (:domain drone)
(:objects 
x0y0z0
x0y0z1
x0y0z2
x0y1z0
x0y1z1
x0y1z2
x1y0z0
x1y0z1
x1y0z2
x1y1z0
x1y1z1
x1y1z2
x2y0z0
x2y0z1
x2y0z2
x2y1z0
x2y1z1
x2y1z2 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 3) 
 (= (min_y) 0)  (= (max_y) 2) 
 (= (min_z) 0)  (= (max_z) 3) 
(= (xl x0y0z0) 0)
(= (yl x0y0z0) 0)
(= (zl x0y0z0) 0)
(= (xl x0y0z1) 0)
(= (yl x0y0z1) 0)
(= (zl x0y0z1) 1)
(= (xl x0y0z2) 0)
(= (yl x0y0z2) 0)
(= (zl x0y0z2) 2)
(= (xl x0y1z0) 0)
(= (yl x0y1z0) 1)
(= (zl x0y1z0) 0)
(= (xl x0y1z1) 0)
(= (yl x0y1z1) 1)
(= (zl x0y1z1) 1)
(= (xl x0y1z2) 0)
(= (yl x0y1z2) 1)
(= (zl x0y1z2) 2)
(= (xl x1y0z0) 1)
(= (yl x1y0z0) 0)
(= (zl x1y0z0) 0)
(= (xl x1y0z1) 1)
(= (yl x1y0z1) 0)
(= (zl x1y0z1) 1)
(= (xl x1y0z2) 1)
(= (yl x1y0z2) 0)
(= (zl x1y0z2) 2)
(= (xl x1y1z0) 1)
(= (yl x1y1z0) 1)
(= (zl x1y1z0) 0)
(= (xl x1y1z1) 1)
(= (yl x1y1z1) 1)
(= (zl x1y1z1) 1)
(= (xl x1y1z2) 1)
(= (yl x1y1z2) 1)
(= (zl x1y1z2) 2)
(= (xl x2y0z0) 2)
(= (yl x2y0z0) 0)
(= (zl x2y0z0) 0)
(= (xl x2y0z1) 2)
(= (yl x2y0z1) 0)
(= (zl x2y0z1) 1)
(= (xl x2y0z2) 2)
(= (yl x2y0z2) 0)
(= (zl x2y0z2) 2)
(= (xl x2y1z0) 2)
(= (yl x2y1z0) 1)
(= (zl x2y1z0) 0)
(= (xl x2y1z1) 2)
(= (yl x2y1z1) 1)
(= (zl x2y1z1) 1)
(= (xl x2y1z2) 2)
(= (yl x2y1z2) 1)
(= (zl x2y1z2) 2)
(= (battery-level) 17)
(= (battery-level-full) 17)
)
(:goal (and 
(visited x0y0z0)
(visited x0y0z1)
(visited x0y0z2)
(visited x0y1z0)
(visited x0y1z1)
(visited x0y1z2)
(visited x1y0z0)
(visited x1y0z1)
(visited x1y0z2)
(visited x1y1z0)
(visited x1y1z1)
(visited x1y1z2)
(visited x2y0z0)
(visited x2y0z1)
(visited x2y0z2)
(visited x2y1z0)
(visited x2y1z1)
(visited x2y1z2)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
