(define (problem mprime-b-1-0-2) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 2) (= (locale f1) 1) (= (locale f2) 3) (= (locale f3) 5) (= (locale f4) 5) (= (locale f5) 2) (= (locale f6) 5) (= (locale f7) 5) (= (locale f8) 4) (= (locale f9) 4) (= (harmony v0) 1) (= (harmony v1) 3) (eats f0 f2) (eats f1 f9) (eats f2 f3) (eats f3 f5) (eats f3 f7) (eats f4 f6) (eats f4 f7) (eats f5 f4) (eats f6 f7) (eats f7 f1) (eats f8 f0) (eats f9 f8) (craves p0 f0) (craves p1 f9) (craves p2 f1) (craves p3 f0) (craves p4 f1) (craves p5 f0) (craves p6 f2) (craves p7 f9) (craves p8 f5) (craves p9 f2) (craves v0 f4) (craves v1 f7))
(:goal (and (craves p3 f5))))
