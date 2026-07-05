;;Instance with 4x8x3 points
(define (problem droneprob_4_8_3_9) (:domain drone)
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
x0y4z0
x0y4z1
x0y4z2
x0y5z0
x0y5z1
x0y5z2
x0y6z0
x0y6z1
x0y6z2
x0y7z0
x0y7z1
x0y7z2
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
x1y4z0
x1y4z1
x1y4z2
x1y5z0
x1y5z1
x1y5z2
x1y6z0
x1y6z1
x1y6z2
x1y7z0
x1y7z1
x1y7z2
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
x2y4z0
x2y4z1
x2y4z2
x2y5z0
x2y5z1
x2y5z2
x2y6z0
x2y6z1
x2y6z2
x2y7z0
x2y7z1
x2y7z2
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
x3y4z0
x3y4z1
x3y4z2
x3y5z0
x3y5z1
x3y5z2
x3y6z0
x3y6z1
x3y6z2
x3y7z0
x3y7z1
x3y7z2 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 4) 
 (= (min_y) 0)  (= (max_y) 8) 
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
(= (xl x0y4z0) 0)
(= (yl x0y4z0) 4)
(= (zl x0y4z0) 0)
(= (xl x0y4z1) 0)
(= (yl x0y4z1) 4)
(= (zl x0y4z1) 1)
(= (xl x0y4z2) 0)
(= (yl x0y4z2) 4)
(= (zl x0y4z2) 2)
(= (xl x0y5z0) 0)
(= (yl x0y5z0) 5)
(= (zl x0y5z0) 0)
(= (xl x0y5z1) 0)
(= (yl x0y5z1) 5)
(= (zl x0y5z1) 1)
(= (xl x0y5z2) 0)
(= (yl x0y5z2) 5)
(= (zl x0y5z2) 2)
(= (xl x0y6z0) 0)
(= (yl x0y6z0) 6)
(= (zl x0y6z0) 0)
(= (xl x0y6z1) 0)
(= (yl x0y6z1) 6)
(= (zl x0y6z1) 1)
(= (xl x0y6z2) 0)
(= (yl x0y6z2) 6)
(= (zl x0y6z2) 2)
(= (xl x0y7z0) 0)
(= (yl x0y7z0) 7)
(= (zl x0y7z0) 0)
(= (xl x0y7z1) 0)
(= (yl x0y7z1) 7)
(= (zl x0y7z1) 1)
(= (xl x0y7z2) 0)
(= (yl x0y7z2) 7)
(= (zl x0y7z2) 2)
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
(= (xl x1y4z0) 1)
(= (yl x1y4z0) 4)
(= (zl x1y4z0) 0)
(= (xl x1y4z1) 1)
(= (yl x1y4z1) 4)
(= (zl x1y4z1) 1)
(= (xl x1y4z2) 1)
(= (yl x1y4z2) 4)
(= (zl x1y4z2) 2)
(= (xl x1y5z0) 1)
(= (yl x1y5z0) 5)
(= (zl x1y5z0) 0)
(= (xl x1y5z1) 1)
(= (yl x1y5z1) 5)
(= (zl x1y5z1) 1)
(= (xl x1y5z2) 1)
(= (yl x1y5z2) 5)
(= (zl x1y5z2) 2)
(= (xl x1y6z0) 1)
(= (yl x1y6z0) 6)
(= (zl x1y6z0) 0)
(= (xl x1y6z1) 1)
(= (yl x1y6z1) 6)
(= (zl x1y6z1) 1)
(= (xl x1y6z2) 1)
(= (yl x1y6z2) 6)
(= (zl x1y6z2) 2)
(= (xl x1y7z0) 1)
(= (yl x1y7z0) 7)
(= (zl x1y7z0) 0)
(= (xl x1y7z1) 1)
(= (yl x1y7z1) 7)
(= (zl x1y7z1) 1)
(= (xl x1y7z2) 1)
(= (yl x1y7z2) 7)
(= (zl x1y7z2) 2)
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
(= (xl x2y4z0) 2)
(= (yl x2y4z0) 4)
(= (zl x2y4z0) 0)
(= (xl x2y4z1) 2)
(= (yl x2y4z1) 4)
(= (zl x2y4z1) 1)
(= (xl x2y4z2) 2)
(= (yl x2y4z2) 4)
(= (zl x2y4z2) 2)
(= (xl x2y5z0) 2)
(= (yl x2y5z0) 5)
(= (zl x2y5z0) 0)
(= (xl x2y5z1) 2)
(= (yl x2y5z1) 5)
(= (zl x2y5z1) 1)
(= (xl x2y5z2) 2)
(= (yl x2y5z2) 5)
(= (zl x2y5z2) 2)
(= (xl x2y6z0) 2)
(= (yl x2y6z0) 6)
(= (zl x2y6z0) 0)
(= (xl x2y6z1) 2)
(= (yl x2y6z1) 6)
(= (zl x2y6z1) 1)
(= (xl x2y6z2) 2)
(= (yl x2y6z2) 6)
(= (zl x2y6z2) 2)
(= (xl x2y7z0) 2)
(= (yl x2y7z0) 7)
(= (zl x2y7z0) 0)
(= (xl x2y7z1) 2)
(= (yl x2y7z1) 7)
(= (zl x2y7z1) 1)
(= (xl x2y7z2) 2)
(= (yl x2y7z2) 7)
(= (zl x2y7z2) 2)
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
(= (xl x3y4z0) 3)
(= (yl x3y4z0) 4)
(= (zl x3y4z0) 0)
(= (xl x3y4z1) 3)
(= (yl x3y4z1) 4)
(= (zl x3y4z1) 1)
(= (xl x3y4z2) 3)
(= (yl x3y4z2) 4)
(= (zl x3y4z2) 2)
(= (xl x3y5z0) 3)
(= (yl x3y5z0) 5)
(= (zl x3y5z0) 0)
(= (xl x3y5z1) 3)
(= (yl x3y5z1) 5)
(= (zl x3y5z1) 1)
(= (xl x3y5z2) 3)
(= (yl x3y5z2) 5)
(= (zl x3y5z2) 2)
(= (xl x3y6z0) 3)
(= (yl x3y6z0) 6)
(= (zl x3y6z0) 0)
(= (xl x3y6z1) 3)
(= (yl x3y6z1) 6)
(= (zl x3y6z1) 1)
(= (xl x3y6z2) 3)
(= (yl x3y6z2) 6)
(= (zl x3y6z2) 2)
(= (xl x3y7z0) 3)
(= (yl x3y7z0) 7)
(= (zl x3y7z0) 0)
(= (xl x3y7z1) 3)
(= (yl x3y7z1) 7)
(= (zl x3y7z1) 1)
(= (xl x3y7z2) 3)
(= (yl x3y7z2) 7)
(= (zl x3y7z2) 2)
(= (battery-level) 31)
(= (battery-level-full) 31)
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
(visited x0y4z0)
(visited x0y4z1)
(visited x0y4z2)
(visited x0y5z0)
(visited x0y5z1)
(visited x0y5z2)
(visited x0y6z0)
(visited x0y6z1)
(visited x0y6z2)
(visited x0y7z0)
(visited x0y7z1)
(visited x0y7z2)
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
(visited x1y4z0)
(visited x1y4z1)
(visited x1y4z2)
(visited x1y5z0)
(visited x1y5z1)
(visited x1y5z2)
(visited x1y6z0)
(visited x1y6z1)
(visited x1y6z2)
(visited x1y7z0)
(visited x1y7z1)
(visited x1y7z2)
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
(visited x2y4z0)
(visited x2y4z1)
(visited x2y4z2)
(visited x2y5z0)
(visited x2y5z1)
(visited x2y5z2)
(visited x2y6z0)
(visited x2y6z1)
(visited x2y6z2)
(visited x2y7z0)
(visited x2y7z1)
(visited x2y7z2)
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
(visited x3y4z0)
(visited x3y4z1)
(visited x3y4z2)
(visited x3y5z0)
(visited x3y5z1)
(visited x3y5z2)
(visited x3y6z0)
(visited x3y6z1)
(visited x3y6z2)
(visited x3y7z0)
(visited x3y7z1)
(visited x3y7z2)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
