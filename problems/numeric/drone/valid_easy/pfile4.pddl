;;Instance with 2x1x1 points
(define (problem droneprob_2_1_1_4) (:domain drone)
(:objects 
x0y0z0
x1y0z0 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 2) 
 (= (min_y) 0)  (= (max_y) 1) 
 (= (min_z) 0)  (= (max_z) 1) 
(= (xl x0y0z0) 0)
(= (yl x0y0z0) 0)
(= (zl x0y0z0) 0)
(= (xl x1y0z0) 1)
(= (yl x1y0z0) 0)
(= (zl x1y0z0) 0)
(= (battery-level) 9)
(= (battery-level-full) 9)
)
(:goal (and 
(visited x0y0z0)
(visited x1y0z0)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
