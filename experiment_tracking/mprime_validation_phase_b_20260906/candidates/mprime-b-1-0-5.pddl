(define (problem mprime-b-1-0-5) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 1) (= (locale f1) 4) (= (locale f2) 1) (= (locale f3) 1) (= (locale f4) 4) (= (locale f5) 1) (= (locale f6) 5) (= (locale f7) 4) (= (locale f8) 5) (= (locale f9) 3) (= (harmony v0) 1) (= (harmony v1) 1) (eats f0 f1) (eats f1 f7) (eats f2 f8) (eats f3 f5) (eats f4 f9) (eats f5 f2) (eats f6 f4) (eats f6 f5) (eats f6 f9) (eats f7 f6) (eats f8 f0) (eats f9 f3) (craves p0 f4) (craves p1 f8) (craves p2 f4) (craves p3 f1) (craves p4 f3) (craves p5 f5) (craves p6 f4) (craves p7 f0) (craves p8 f2) (craves p9 f5) (craves v0 f2) (craves v1 f0))
(:goal (and (craves p6 f6))))
