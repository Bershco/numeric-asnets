;;Instance with 1x2x2 points
(define (problem droneprob_1_2_2_7) (:domain drone)
(:objects 
x0y0z0
x0y0z1
x0y1z0
x0y1z1 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 1) 
 (= (min_y) 0)  (= (max_y) 2) 
 (= (min_z) 0)  (= (max_z) 2) 
(= (xl x0y0z0) 0)
(= (yl x0y0z0) 0)
(= (zl x0y0z0) 0)
(= (xl x0y0z1) 0)
(= (yl x0y0z1) 0)
(= (zl x0y0z1) 1)
(= (xl x0y1z0) 0)
(= (yl x0y1z0) 1)
(= (zl x0y1z0) 0)
(= (xl x0y1z1) 0)
(= (yl x0y1z1) 1)
(= (zl x0y1z1) 1)
(= (battery-level) 11)
(= (battery-level-full) 11)
)
(:goal (and 
(visited x0y0z0)
(visited x0y0z1)
(visited x0y1z0)
(visited x0y1z1)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
