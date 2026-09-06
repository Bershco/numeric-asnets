(define (problem mprime-b-1-0-26) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 1) (= (locale f1) 1) (= (locale f2) 2) (= (locale f3) 2) (= (locale f4) 4) (= (locale f5) 1) (= (locale f6) 3) (= (locale f7) 1) (= (locale f8) 3) (= (locale f9) 3) (= (harmony v0) 2) (= (harmony v1) 2) (eats f0 f1) (eats f1 f9) (eats f2 f6) (eats f2 f7) (eats f3 f4) (eats f4 f8) (eats f5 f3) (eats f6 f0) (eats f7 f5) (eats f8 f6) (eats f9 f2) (craves p0 f6) (craves p1 f7) (craves p2 f8) (craves p3 f1) (craves p4 f1) (craves p5 f9) (craves p6 f9) (craves p7 f0) (craves p8 f3) (craves p9 f6) (craves v0 f6) (craves v1 f4))
(:goal (and (craves p0 f5))))
