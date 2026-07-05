;;Instance with 5x4x4 points
(define (problem droneprob_5_4_4_0) (:domain drone)
(:objects 
x0y0z0
x0y0z1
x0y0z2
x0y0z3
x0y1z0
x0y1z1
x0y1z2
x0y1z3
x0y2z0
x0y2z1
x0y2z2
x0y2z3
x0y3z0
x0y3z1
x0y3z2
x0y3z3
x1y0z0
x1y0z1
x1y0z2
x1y0z3
x1y1z0
x1y1z1
x1y1z2
x1y1z3
x1y2z0
x1y2z1
x1y2z2
x1y2z3
x1y3z0
x1y3z1
x1y3z2
x1y3z3
x2y0z0
x2y0z1
x2y0z2
x2y0z3
x2y1z0
x2y1z1
x2y1z2
x2y1z3
x2y2z0
x2y2z1
x2y2z2
x2y2z3
x2y3z0
x2y3z1
x2y3z2
x2y3z3
x3y0z0
x3y0z1
x3y0z2
x3y0z3
x3y1z0
x3y1z1
x3y1z2
x3y1z3
x3y2z0
x3y2z1
x3y2z2
x3y2z3
x3y3z0
x3y3z1
x3y3z2
x3y3z3
x4y0z0
x4y0z1
x4y0z2
x4y0z3
x4y1z0
x4y1z1
x4y1z2
x4y1z3
x4y2z0
x4y2z1
x4y2z2
x4y2z3
x4y3z0
x4y3z1
x4y3z2
x4y3z3 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 5) 
 (= (min_y) 0)  (= (max_y) 4) 
 (= (min_z) 0)  (= (max_z) 4) 
(= (xl x0y0z0) 0)
(= (yl x0y0z0) 0)
(= (zl x0y0z0) 0)
(= (xl x0y0z1) 0)
(= (yl x0y0z1) 0)
(= (zl x0y0z1) 1)
(= (xl x0y0z2) 0)
(= (yl x0y0z2) 0)
(= (zl x0y0z2) 2)
(= (xl x0y0z3) 0)
(= (yl x0y0z3) 0)
(= (zl x0y0z3) 3)
(= (xl x0y1z0) 0)
(= (yl x0y1z0) 1)
(= (zl x0y1z0) 0)
(= (xl x0y1z1) 0)
(= (yl x0y1z1) 1)
(= (zl x0y1z1) 1)
(= (xl x0y1z2) 0)
(= (yl x0y1z2) 1)
(= (zl x0y1z2) 2)
(= (xl x0y1z3) 0)
(= (yl x0y1z3) 1)
(= (zl x0y1z3) 3)
(= (xl x0y2z0) 0)
(= (yl x0y2z0) 2)
(= (zl x0y2z0) 0)
(= (xl x0y2z1) 0)
(= (yl x0y2z1) 2)
(= (zl x0y2z1) 1)
(= (xl x0y2z2) 0)
(= (yl x0y2z2) 2)
(= (zl x0y2z2) 2)
(= (xl x0y2z3) 0)
(= (yl x0y2z3) 2)
(= (zl x0y2z3) 3)
(= (xl x0y3z0) 0)
(= (yl x0y3z0) 3)
(= (zl x0y3z0) 0)
(= (xl x0y3z1) 0)
(= (yl x0y3z1) 3)
(= (zl x0y3z1) 1)
(= (xl x0y3z2) 0)
(= (yl x0y3z2) 3)
(= (zl x0y3z2) 2)
(= (xl x0y3z3) 0)
(= (yl x0y3z3) 3)
(= (zl x0y3z3) 3)
(= (xl x1y0z0) 1)
(= (yl x1y0z0) 0)
(= (zl x1y0z0) 0)
(= (xl x1y0z1) 1)
(= (yl x1y0z1) 0)
(= (zl x1y0z1) 1)
(= (xl x1y0z2) 1)
(= (yl x1y0z2) 0)
(= (zl x1y0z2) 2)
(= (xl x1y0z3) 1)
(= (yl x1y0z3) 0)
(= (zl x1y0z3) 3)
(= (xl x1y1z0) 1)
(= (yl x1y1z0) 1)
(= (zl x1y1z0) 0)
(= (xl x1y1z1) 1)
(= (yl x1y1z1) 1)
(= (zl x1y1z1) 1)
(= (xl x1y1z2) 1)
(= (yl x1y1z2) 1)
(= (zl x1y1z2) 2)
(= (xl x1y1z3) 1)
(= (yl x1y1z3) 1)
(= (zl x1y1z3) 3)
(= (xl x1y2z0) 1)
(= (yl x1y2z0) 2)
(= (zl x1y2z0) 0)
(= (xl x1y2z1) 1)
(= (yl x1y2z1) 2)
(= (zl x1y2z1) 1)
(= (xl x1y2z2) 1)
(= (yl x1y2z2) 2)
(= (zl x1y2z2) 2)
(= (xl x1y2z3) 1)
(= (yl x1y2z3) 2)
(= (zl x1y2z3) 3)
(= (xl x1y3z0) 1)
(= (yl x1y3z0) 3)
(= (zl x1y3z0) 0)
(= (xl x1y3z1) 1)
(= (yl x1y3z1) 3)
(= (zl x1y3z1) 1)
(= (xl x1y3z2) 1)
(= (yl x1y3z2) 3)
(= (zl x1y3z2) 2)
(= (xl x1y3z3) 1)
(= (yl x1y3z3) 3)
(= (zl x1y3z3) 3)
(= (xl x2y0z0) 2)
(= (yl x2y0z0) 0)
(= (zl x2y0z0) 0)
(= (xl x2y0z1) 2)
(= (yl x2y0z1) 0)
(= (zl x2y0z1) 1)
(= (xl x2y0z2) 2)
(= (yl x2y0z2) 0)
(= (zl x2y0z2) 2)
(= (xl x2y0z3) 2)
(= (yl x2y0z3) 0)
(= (zl x2y0z3) 3)
(= (xl x2y1z0) 2)
(= (yl x2y1z0) 1)
(= (zl x2y1z0) 0)
(= (xl x2y1z1) 2)
(= (yl x2y1z1) 1)
(= (zl x2y1z1) 1)
(= (xl x2y1z2) 2)
(= (yl x2y1z2) 1)
(= (zl x2y1z2) 2)
(= (xl x2y1z3) 2)
(= (yl x2y1z3) 1)
(= (zl x2y1z3) 3)
(= (xl x2y2z0) 2)
(= (yl x2y2z0) 2)
(= (zl x2y2z0) 0)
(= (xl x2y2z1) 2)
(= (yl x2y2z1) 2)
(= (zl x2y2z1) 1)
(= (xl x2y2z2) 2)
(= (yl x2y2z2) 2)
(= (zl x2y2z2) 2)
(= (xl x2y2z3) 2)
(= (yl x2y2z3) 2)
(= (zl x2y2z3) 3)
(= (xl x2y3z0) 2)
(= (yl x2y3z0) 3)
(= (zl x2y3z0) 0)
(= (xl x2y3z1) 2)
(= (yl x2y3z1) 3)
(= (zl x2y3z1) 1)
(= (xl x2y3z2) 2)
(= (yl x2y3z2) 3)
(= (zl x2y3z2) 2)
(= (xl x2y3z3) 2)
(= (yl x2y3z3) 3)
(= (zl x2y3z3) 3)
(= (xl x3y0z0) 3)
(= (yl x3y0z0) 0)
(= (zl x3y0z0) 0)
(= (xl x3y0z1) 3)
(= (yl x3y0z1) 0)
(= (zl x3y0z1) 1)
(= (xl x3y0z2) 3)
(= (yl x3y0z2) 0)
(= (zl x3y0z2) 2)
(= (xl x3y0z3) 3)
(= (yl x3y0z3) 0)
(= (zl x3y0z3) 3)
(= (xl x3y1z0) 3)
(= (yl x3y1z0) 1)
(= (zl x3y1z0) 0)
(= (xl x3y1z1) 3)
(= (yl x3y1z1) 1)
(= (zl x3y1z1) 1)
(= (xl x3y1z2) 3)
(= (yl x3y1z2) 1)
(= (zl x3y1z2) 2)
(= (xl x3y1z3) 3)
(= (yl x3y1z3) 1)
(= (zl x3y1z3) 3)
(= (xl x3y2z0) 3)
(= (yl x3y2z0) 2)
(= (zl x3y2z0) 0)
(= (xl x3y2z1) 3)
(= (yl x3y2z1) 2)
(= (zl x3y2z1) 1)
(= (xl x3y2z2) 3)
(= (yl x3y2z2) 2)
(= (zl x3y2z2) 2)
(= (xl x3y2z3) 3)
(= (yl x3y2z3) 2)
(= (zl x3y2z3) 3)
(= (xl x3y3z0) 3)
(= (yl x3y3z0) 3)
(= (zl x3y3z0) 0)
(= (xl x3y3z1) 3)
(= (yl x3y3z1) 3)
(= (zl x3y3z1) 1)
(= (xl x3y3z2) 3)
(= (yl x3y3z2) 3)
(= (zl x3y3z2) 2)
(= (xl x3y3z3) 3)
(= (yl x3y3z3) 3)
(= (zl x3y3z3) 3)
(= (xl x4y0z0) 4)
(= (yl x4y0z0) 0)
(= (zl x4y0z0) 0)
(= (xl x4y0z1) 4)
(= (yl x4y0z1) 0)
(= (zl x4y0z1) 1)
(= (xl x4y0z2) 4)
(= (yl x4y0z2) 0)
(= (zl x4y0z2) 2)
(= (xl x4y0z3) 4)
(= (yl x4y0z3) 0)
(= (zl x4y0z3) 3)
(= (xl x4y1z0) 4)
(= (yl x4y1z0) 1)
(= (zl x4y1z0) 0)
(= (xl x4y1z1) 4)
(= (yl x4y1z1) 1)
(= (zl x4y1z1) 1)
(= (xl x4y1z2) 4)
(= (yl x4y1z2) 1)
(= (zl x4y1z2) 2)
(= (xl x4y1z3) 4)
(= (yl x4y1z3) 1)
(= (zl x4y1z3) 3)
(= (xl x4y2z0) 4)
(= (yl x4y2z0) 2)
(= (zl x4y2z0) 0)
(= (xl x4y2z1) 4)
(= (yl x4y2z1) 2)
(= (zl x4y2z1) 1)
(= (xl x4y2z2) 4)
(= (yl x4y2z2) 2)
(= (zl x4y2z2) 2)
(= (xl x4y2z3) 4)
(= (yl x4y2z3) 2)
(= (zl x4y2z3) 3)
(= (xl x4y3z0) 4)
(= (yl x4y3z0) 3)
(= (zl x4y3z0) 0)
(= (xl x4y3z1) 4)
(= (yl x4y3z1) 3)
(= (zl x4y3z1) 1)
(= (xl x4y3z2) 4)
(= (yl x4y3z2) 3)
(= (zl x4y3z2) 2)
(= (xl x4y3z3) 4)
(= (yl x4y3z3) 3)
(= (zl x4y3z3) 3)
(= (battery-level) 27)
(= (battery-level-full) 27)
)
(:goal (and 
(visited x0y0z0)
(visited x0y0z1)
(visited x0y0z2)
(visited x0y0z3)
(visited x0y1z0)
(visited x0y1z1)
(visited x0y1z2)
(visited x0y1z3)
(visited x0y2z0)
(visited x0y2z1)
(visited x0y2z2)
(visited x0y2z3)
(visited x0y3z0)
(visited x0y3z1)
(visited x0y3z2)
(visited x0y3z3)
(visited x1y0z0)
(visited x1y0z1)
(visited x1y0z2)
(visited x1y0z3)
(visited x1y1z0)
(visited x1y1z1)
(visited x1y1z2)
(visited x1y1z3)
(visited x1y2z0)
(visited x1y2z1)
(visited x1y2z2)
(visited x1y2z3)
(visited x1y3z0)
(visited x1y3z1)
(visited x1y3z2)
(visited x1y3z3)
(visited x2y0z0)
(visited x2y0z1)
(visited x2y0z2)
(visited x2y0z3)
(visited x2y1z0)
(visited x2y1z1)
(visited x2y1z2)
(visited x2y1z3)
(visited x2y2z0)
(visited x2y2z1)
(visited x2y2z2)
(visited x2y2z3)
(visited x2y3z0)
(visited x2y3z1)
(visited x2y3z2)
(visited x2y3z3)
(visited x3y0z0)
(visited x3y0z1)
(visited x3y0z2)
(visited x3y0z3)
(visited x3y1z0)
(visited x3y1z1)
(visited x3y1z2)
(visited x3y1z3)
(visited x3y2z0)
(visited x3y2z1)
(visited x3y2z2)
(visited x3y2z3)
(visited x3y3z0)
(visited x3y3z1)
(visited x3y3z2)
(visited x3y3z3)
(visited x4y0z0)
(visited x4y0z1)
(visited x4y0z2)
(visited x4y0z3)
(visited x4y1z0)
(visited x4y1z1)
(visited x4y1z2)
(visited x4y1z3)
(visited x4y2z0)
(visited x4y2z1)
(visited x4y2z2)
(visited x4y2z3)
(visited x4y3z0)
(visited x4y3z1)
(visited x4y3z2)
(visited x4y3z3)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
