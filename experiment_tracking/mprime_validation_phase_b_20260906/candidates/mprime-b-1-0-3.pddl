(define (problem mprime-b-1-0-3) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 5) (= (locale f1) 2) (= (locale f2) 3) (= (locale f3) 3) (= (locale f4) 4) (= (locale f5) 4) (= (locale f6) 2) (= (locale f7) 5) (= (locale f8) 4) (= (locale f9) 1) (= (harmony v0) 3) (= (harmony v1) 2) (eats f0 f4) (eats f1 f2) (eats f2 f3) (eats f3 f8) (eats f4 f3) (eats f4 f9) (eats f5 f6) (eats f6 f7) (eats f7 f1) (eats f8 f0) (eats f8 f9) (eats f9 f5) (craves p0 f2) (craves p1 f7) (craves p2 f6) (craves p3 f8) (craves p4 f5) (craves p5 f3) (craves p6 f5) (craves p7 f8) (craves p8 f9) (craves p9 f0) (craves v0 f7) (craves v1 f1))
(:goal (and (craves p8 f5))))
