(define (problem mprime-b-1-0-35) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 1) (= (locale f1) 5) (= (locale f2) 2) (= (locale f3) 5) (= (locale f4) 3) (= (locale f5) 5) (= (locale f6) 1) (= (locale f7) 1) (= (locale f8) 4) (= (locale f9) 1) (= (harmony v0) 1) (= (harmony v1) 1) (eats f0 f5) (eats f1 f9) (eats f2 f0) (eats f3 f1) (eats f4 f0) (eats f4 f8) (eats f5 f6) (eats f6 f3) (eats f7 f2) (eats f7 f9) (eats f8 f7) (eats f9 f4) (craves p0 f0) (craves p1 f2) (craves p2 f9) (craves p3 f2) (craves p4 f6) (craves p5 f0) (craves p6 f7) (craves p7 f4) (craves p8 f1) (craves p9 f4) (craves v0 f0) (craves v1 f6))
(:goal (and (craves p9 f2))))
