(define (problem mprime-b-1-0-24) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 2) (= (locale f1) 4) (= (locale f2) 4) (= (locale f3) 2) (= (locale f4) 2) (= (locale f5) 3) (= (locale f6) 2) (= (locale f7) 2) (= (locale f8) 2) (= (locale f9) 2) (= (harmony v0) 1) (= (harmony v1) 2) (eats f0 f2) (eats f0 f9) (eats f1 f7) (eats f2 f3) (eats f3 f4) (eats f3 f6) (eats f4 f1) (eats f5 f0) (eats f6 f8) (eats f7 f6) (eats f8 f5) (eats f9 f2) (craves p0 f7) (craves p1 f6) (craves p2 f4) (craves p3 f0) (craves p4 f5) (craves p5 f8) (craves p6 f8) (craves p7 f5) (craves p8 f0) (craves p9 f4) (craves v0 f5) (craves v1 f6))
(:goal (and (craves p0 f1))))
