(define (problem mprime-b-1-0-9) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 5) (= (locale f1) 2) (= (locale f2) 4) (= (locale f3) 4) (= (locale f4) 2) (= (locale f5) 1) (= (locale f6) 5) (= (locale f7) 2) (= (locale f8) 5) (= (locale f9) 1) (= (harmony v0) 2) (= (harmony v1) 3) (eats f0 f5) (eats f1 f6) (eats f2 f4) (eats f3 f7) (eats f4 f1) (eats f5 f3) (eats f6 f8) (eats f6 f9) (eats f7 f2) (eats f7 f6) (eats f8 f9) (eats f9 f0) (craves p0 f1) (craves p1 f1) (craves p2 f1) (craves p3 f2) (craves p4 f0) (craves p5 f5) (craves p6 f9) (craves p7 f4) (craves p8 f6) (craves p9 f2) (craves v0 f0) (craves v1 f0))
(:goal (and (craves p4 f8))))
