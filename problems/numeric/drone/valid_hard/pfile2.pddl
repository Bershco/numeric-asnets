;;Instance with 7x4x3 points
(define (problem droneprob_7_4_3_2) (:domain drone)
(:objects 
x0y0z0
x0y0z1
x0y0z2
x0y1z0
x0y1z1
x0y1z2
x0y2z0
x0y2z1
x0y2z2
x0y3z0
x0y3z1
x0y3z2
x1y0z0
x1y0z1
x1y0z2
x1y1z0
x1y1z1
x1y1z2
x1y2z0
x1y2z1
x1y2z2
x1y3z0
x1y3z1
x1y3z2
x2y0z0
x2y0z1
x2y0z2
x2y1z0
x2y1z1
x2y1z2
x2y2z0
x2y2z1
x2y2z2
x2y3z0
x2y3z1
x2y3z2
x3y0z0
x3y0z1
x3y0z2
x3y1z0
x3y1z1
x3y1z2
x3y2z0
x3y2z1
x3y2z2
x3y3z0
x3y3z1
x3y3z2
x4y0z0
x4y0z1
x4y0z2
x4y1z0
x4y1z1
x4y1z2
x4y2z0
x4y2z1
x4y2z2
x4y3z0
x4y3z1
x4y3z2
x5y0z0
x5y0z1
x5y0z2
x5y1z0
x5y1z1
x5y1z2
x5y2z0
x5y2z1
x5y2z2
x5y3z0
x5y3z1
x5y3z2
x6y0z0
x6y0z1
x6y0z2
x6y1z0
x6y1z1
x6y1z2
x6y2z0
x6y2z1
x6y2z2
x6y3z0
x6y3z1
x6y3z2 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 7) 
 (= (min_y) 0)  (= (max_y) 4) 
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
(= (xl x0y2z0) 0)
(= (yl x0y2z0) 2)
(= (zl x0y2z0) 0)
(= (xl x0y2z1) 0)
(= (yl x0y2z1) 2)
(= (zl x0y2z1) 1)
(= (xl x0y2z2) 0)
(= (yl x0y2z2) 2)
(= (zl x0y2z2) 2)
(= (xl x0y3z0) 0)
(= (yl x0y3z0) 3)
(= (zl x0y3z0) 0)
(= (xl x0y3z1) 0)
(= (yl x0y3z1) 3)
(= (zl x0y3z1) 1)
(= (xl x0y3z2) 0)
(= (yl x0y3z2) 3)
(= (zl x0y3z2) 2)
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
(= (xl x1y2z0) 1)
(= (yl x1y2z0) 2)
(= (zl x1y2z0) 0)
(= (xl x1y2z1) 1)
(= (yl x1y2z1) 2)
(= (zl x1y2z1) 1)
(= (xl x1y2z2) 1)
(= (yl x1y2z2) 2)
(= (zl x1y2z2) 2)
(= (xl x1y3z0) 1)
(= (yl x1y3z0) 3)
(= (zl x1y3z0) 0)
(= (xl x1y3z1) 1)
(= (yl x1y3z1) 3)
(= (zl x1y3z1) 1)
(= (xl x1y3z2) 1)
(= (yl x1y3z2) 3)
(= (zl x1y3z2) 2)
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
(= (xl x2y2z0) 2)
(= (yl x2y2z0) 2)
(= (zl x2y2z0) 0)
(= (xl x2y2z1) 2)
(= (yl x2y2z1) 2)
(= (zl x2y2z1) 1)
(= (xl x2y2z2) 2)
(= (yl x2y2z2) 2)
(= (zl x2y2z2) 2)
(= (xl x2y3z0) 2)
(= (yl x2y3z0) 3)
(= (zl x2y3z0) 0)
(= (xl x2y3z1) 2)
(= (yl x2y3z1) 3)
(= (zl x2y3z1) 1)
(= (xl x2y3z2) 2)
(= (yl x2y3z2) 3)
(= (zl x2y3z2) 2)
(= (xl x3y0z0) 3)
(= (yl x3y0z0) 0)
(= (zl x3y0z0) 0)
(= (xl x3y0z1) 3)
(= (yl x3y0z1) 0)
(= (zl x3y0z1) 1)
(= (xl x3y0z2) 3)
(= (yl x3y0z2) 0)
(= (zl x3y0z2) 2)
(= (xl x3y1z0) 3)
(= (yl x3y1z0) 1)
(= (zl x3y1z0) 0)
(= (xl x3y1z1) 3)
(= (yl x3y1z1) 1)
(= (zl x3y1z1) 1)
(= (xl x3y1z2) 3)
(= (yl x3y1z2) 1)
(= (zl x3y1z2) 2)
(= (xl x3y2z0) 3)
(= (yl x3y2z0) 2)
(= (zl x3y2z0) 0)
(= (xl x3y2z1) 3)
(= (yl x3y2z1) 2)
(= (zl x3y2z1) 1)
(= (xl x3y2z2) 3)
(= (yl x3y2z2) 2)
(= (zl x3y2z2) 2)
(= (xl x3y3z0) 3)
(= (yl x3y3z0) 3)
(= (zl x3y3z0) 0)
(= (xl x3y3z1) 3)
(= (yl x3y3z1) 3)
(= (zl x3y3z1) 1)
(= (xl x3y3z2) 3)
(= (yl x3y3z2) 3)
(= (zl x3y3z2) 2)
(= (xl x4y0z0) 4)
(= (yl x4y0z0) 0)
(= (zl x4y0z0) 0)
(= (xl x4y0z1) 4)
(= (yl x4y0z1) 0)
(= (zl x4y0z1) 1)
(= (xl x4y0z2) 4)
(= (yl x4y0z2) 0)
(= (zl x4y0z2) 2)
(= (xl x4y1z0) 4)
(= (yl x4y1z0) 1)
(= (zl x4y1z0) 0)
(= (xl x4y1z1) 4)
(= (yl x4y1z1) 1)
(= (zl x4y1z1) 1)
(= (xl x4y1z2) 4)
(= (yl x4y1z2) 1)
(= (zl x4y1z2) 2)
(= (xl x4y2z0) 4)
(= (yl x4y2z0) 2)
(= (zl x4y2z0) 0)
(= (xl x4y2z1) 4)
(= (yl x4y2z1) 2)
(= (zl x4y2z1) 1)
(= (xl x4y2z2) 4)
(= (yl x4y2z2) 2)
(= (zl x4y2z2) 2)
(= (xl x4y3z0) 4)
(= (yl x4y3z0) 3)
(= (zl x4y3z0) 0)
(= (xl x4y3z1) 4)
(= (yl x4y3z1) 3)
(= (zl x4y3z1) 1)
(= (xl x4y3z2) 4)
(= (yl x4y3z2) 3)
(= (zl x4y3z2) 2)
(= (xl x5y0z0) 5)
(= (yl x5y0z0) 0)
(= (zl x5y0z0) 0)
(= (xl x5y0z1) 5)
(= (yl x5y0z1) 0)
(= (zl x5y0z1) 1)
(= (xl x5y0z2) 5)
(= (yl x5y0z2) 0)
(= (zl x5y0z2) 2)
(= (xl x5y1z0) 5)
(= (yl x5y1z0) 1)
(= (zl x5y1z0) 0)
(= (xl x5y1z1) 5)
(= (yl x5y1z1) 1)
(= (zl x5y1z1) 1)
(= (xl x5y1z2) 5)
(= (yl x5y1z2) 1)
(= (zl x5y1z2) 2)
(= (xl x5y2z0) 5)
(= (yl x5y2z0) 2)
(= (zl x5y2z0) 0)
(= (xl x5y2z1) 5)
(= (yl x5y2z1) 2)
(= (zl x5y2z1) 1)
(= (xl x5y2z2) 5)
(= (yl x5y2z2) 2)
(= (zl x5y2z2) 2)
(= (xl x5y3z0) 5)
(= (yl x5y3z0) 3)
(= (zl x5y3z0) 0)
(= (xl x5y3z1) 5)
(= (yl x5y3z1) 3)
(= (zl x5y3z1) 1)
(= (xl x5y3z2) 5)
(= (yl x5y3z2) 3)
(= (zl x5y3z2) 2)
(= (xl x6y0z0) 6)
(= (yl x6y0z0) 0)
(= (zl x6y0z0) 0)
(= (xl x6y0z1) 6)
(= (yl x6y0z1) 0)
(= (zl x6y0z1) 1)
(= (xl x6y0z2) 6)
(= (yl x6y0z2) 0)
(= (zl x6y0z2) 2)
(= (xl x6y1z0) 6)
(= (yl x6y1z0) 1)
(= (zl x6y1z0) 0)
(= (xl x6y1z1) 6)
(= (yl x6y1z1) 1)
(= (zl x6y1z1) 1)
(= (xl x6y1z2) 6)
(= (yl x6y1z2) 1)
(= (zl x6y1z2) 2)
(= (xl x6y2z0) 6)
(= (yl x6y2z0) 2)
(= (zl x6y2z0) 0)
(= (xl x6y2z1) 6)
(= (yl x6y2z1) 2)
(= (zl x6y2z1) 1)
(= (xl x6y2z2) 6)
(= (yl x6y2z2) 2)
(= (zl x6y2z2) 2)
(= (xl x6y3z0) 6)
(= (yl x6y3z0) 3)
(= (zl x6y3z0) 0)
(= (xl x6y3z1) 6)
(= (yl x6y3z1) 3)
(= (zl x6y3z1) 1)
(= (xl x6y3z2) 6)
(= (yl x6y3z2) 3)
(= (zl x6y3z2) 2)
(= (battery-level) 29)
(= (battery-level-full) 29)
)
(:goal (and 
(visited x0y0z0)
(visited x0y0z1)
(visited x0y0z2)
(visited x0y1z0)
(visited x0y1z1)
(visited x0y1z2)
(visited x0y2z0)
(visited x0y2z1)
(visited x0y2z2)
(visited x0y3z0)
(visited x0y3z1)
(visited x0y3z2)
(visited x1y0z0)
(visited x1y0z1)
(visited x1y0z2)
(visited x1y1z0)
(visited x1y1z1)
(visited x1y1z2)
(visited x1y2z0)
(visited x1y2z1)
(visited x1y2z2)
(visited x1y3z0)
(visited x1y3z1)
(visited x1y3z2)
(visited x2y0z0)
(visited x2y0z1)
(visited x2y0z2)
(visited x2y1z0)
(visited x2y1z1)
(visited x2y1z2)
(visited x2y2z0)
(visited x2y2z1)
(visited x2y2z2)
(visited x2y3z0)
(visited x2y3z1)
(visited x2y3z2)
(visited x3y0z0)
(visited x3y0z1)
(visited x3y0z2)
(visited x3y1z0)
(visited x3y1z1)
(visited x3y1z2)
(visited x3y2z0)
(visited x3y2z1)
(visited x3y2z2)
(visited x3y3z0)
(visited x3y3z1)
(visited x3y3z2)
(visited x4y0z0)
(visited x4y0z1)
(visited x4y0z2)
(visited x4y1z0)
(visited x4y1z1)
(visited x4y1z2)
(visited x4y2z0)
(visited x4y2z1)
(visited x4y2z2)
(visited x4y3z0)
(visited x4y3z1)
(visited x4y3z2)
(visited x5y0z0)
(visited x5y0z1)
(visited x5y0z2)
(visited x5y1z0)
(visited x5y1z1)
(visited x5y1z2)
(visited x5y2z0)
(visited x5y2z1)
(visited x5y2z2)
(visited x5y3z0)
(visited x5y3z1)
(visited x5y3z2)
(visited x6y0z0)
(visited x6y0z1)
(visited x6y0z2)
(visited x6y1z0)
(visited x6y1z1)
(visited x6y1z2)
(visited x6y2z0)
(visited x6y2z1)
(visited x6y2z2)
(visited x6y3z0)
(visited x6y3z1)
(visited x6y3z2)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
