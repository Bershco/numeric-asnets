(define (problem mprime-b-0-0-1) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 3) (= (locale f1) 5) (= (locale f2) 5) (= (locale f3) 2) (= (locale f4) 3) (= (locale f5) 5) (= (locale f6) 3) (= (locale f7) 2) (= (locale f8) 3) (= (locale f9) 2) (= (harmony v0) 1) (= (harmony v1) 2) (eats f0 f3) (eats f1 f5) (eats f2 f0) (eats f3 f4) (eats f4 f6) (eats f5 f7) (eats f5 f8) (eats f6 f9) (eats f7 f2) (eats f8 f1) (eats f9 f6) (eats f9 f8) (craves p0 f5) (craves p1 f0) (craves p2 f7) (craves p3 f2) (craves p4 f6) (craves p5 f6) (craves p6 f5) (craves p7 f8) (craves p8 f3) (craves p9 f7) (craves v0 f1) (craves v1 f5))
(:goal (and (craves p8 f4))))
