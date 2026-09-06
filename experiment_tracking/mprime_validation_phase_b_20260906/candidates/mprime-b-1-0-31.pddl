(define (problem mprime-b-1-0-31) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 1) (= (locale f1) 5) (= (locale f2) 4) (= (locale f3) 2) (= (locale f4) 5) (= (locale f5) 3) (= (locale f6) 1) (= (locale f7) 1) (= (locale f8) 5) (= (locale f9) 2) (= (harmony v0) 3) (= (harmony v1) 1) (eats f0 f1) (eats f0 f6) (eats f1 f2) (eats f2 f5) (eats f3 f7) (eats f4 f8) (eats f5 f0) (eats f6 f9) (eats f7 f1) (eats f8 f1) (eats f8 f3) (eats f9 f4) (craves p0 f6) (craves p1 f7) (craves p2 f0) (craves p3 f7) (craves p4 f2) (craves p5 f5) (craves p6 f0) (craves p7 f9) (craves p8 f6) (craves p9 f3) (craves v0 f5) (craves v1 f9))
(:goal (and (craves p4 f7))))
