(define (problem mprime-b-1-0-12) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 1) (= (locale f1) 1) (= (locale f2) 5) (= (locale f3) 1) (= (locale f4) 5) (= (locale f5) 4) (= (locale f6) 2) (= (locale f7) 4) (= (locale f8) 1) (= (locale f9) 2) (= (harmony v0) 1) (= (harmony v1) 2) (eats f0 f2) (eats f1 f5) (eats f2 f4) (eats f3 f9) (eats f4 f1) (eats f5 f3) (eats f6 f5) (eats f6 f8) (eats f7 f6) (eats f8 f0) (eats f9 f7) (craves p0 f2) (craves p1 f9) (craves p2 f4) (craves p3 f3) (craves p4 f4) (craves p5 f5) (craves p6 f6) (craves p7 f5) (craves p8 f8) (craves p9 f3) (craves v0 f8) (craves v1 f9))
(:goal (and (craves p0 f1))))
