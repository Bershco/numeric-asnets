(define (problem mprime-b-1-0-4) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 1) (= (locale f1) 1) (= (locale f2) 5) (= (locale f3) 5) (= (locale f4) 3) (= (locale f5) 3) (= (locale f6) 2) (= (locale f7) 5) (= (locale f8) 1) (= (locale f9) 4) (= (harmony v0) 3) (= (harmony v1) 3) (eats f0 f5) (eats f1 f0) (eats f2 f1) (eats f3 f4) (eats f4 f5) (eats f4 f6) (eats f5 f7) (eats f6 f1) (eats f6 f2) (eats f7 f9) (eats f8 f3) (eats f9 f8) (craves p0 f5) (craves p1 f2) (craves p2 f3) (craves p3 f5) (craves p4 f5) (craves p5 f3) (craves p6 f8) (craves p7 f4) (craves p8 f7) (craves p9 f6) (craves v0 f9) (craves v1 f5))
(:goal (and (craves p5 f4))))
