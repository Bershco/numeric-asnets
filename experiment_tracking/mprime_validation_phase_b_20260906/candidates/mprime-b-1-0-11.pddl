(define (problem mprime-b-1-0-11) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 4) (= (locale f1) 4) (= (locale f2) 1) (= (locale f3) 1) (= (locale f4) 1) (= (locale f5) 5) (= (locale f6) 1) (= (locale f7) 1) (= (locale f8) 3) (= (locale f9) 5) (= (harmony v0) 2) (= (harmony v1) 1) (eats f0 f7) (eats f1 f9) (eats f2 f3) (eats f2 f6) (eats f3 f5) (eats f4 f5) (eats f4 f8) (eats f5 f6) (eats f6 f4) (eats f7 f2) (eats f8 f1) (eats f9 f0) (craves p0 f7) (craves p1 f8) (craves p2 f1) (craves p3 f7) (craves p4 f7) (craves p5 f0) (craves p6 f5) (craves p7 f7) (craves p8 f2) (craves p9 f3) (craves v0 f1) (craves v1 f5))
(:goal (and (craves p1 f7))))
