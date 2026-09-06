(define (problem mprime-b-1-0-6) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 4) (= (locale f1) 4) (= (locale f2) 1) (= (locale f3) 4) (= (locale f4) 2) (= (locale f5) 4) (= (locale f6) 2) (= (locale f7) 5) (= (locale f8) 2) (= (locale f9) 1) (= (harmony v0) 3) (= (harmony v1) 1) (eats f0 f1) (eats f1 f6) (eats f2 f7) (eats f3 f2) (eats f4 f1) (eats f4 f8) (eats f5 f0) (eats f6 f4) (eats f7 f9) (eats f8 f3) (eats f9 f5) (craves p0 f3) (craves p1 f9) (craves p2 f5) (craves p3 f5) (craves p4 f1) (craves p5 f9) (craves p6 f5) (craves p7 f2) (craves p8 f5) (craves p9 f8) (craves v0 f2) (craves v1 f0))
(:goal (and (craves p2 f6))))
