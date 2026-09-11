(define (problem mprime-c-0-22) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 - food v0 v1 - pleasure p0 p1 p2 p3 p4 - pain)
(:init (= (locale f0) 4) (= (locale f1) 4) (= (locale f2) 2) (= (locale f3) 8) (= (locale f4) 1) (= (locale f5) 9) (= (locale f6) 6) (= (locale f7) 4) (= (locale f8) 1) (= (harmony v0) 2) (= (harmony v1) 3) (eats f0 f4) (eats f1 f0) (eats f1 f3) (eats f1 f4) (eats f2 f0) (eats f2 f1) (eats f2 f6) (eats f2 f7) (eats f3 f0) (eats f3 f2) (eats f3 f7) (eats f4 f5) (eats f4 f7) (eats f5 f7) (eats f5 f8) (eats f6 f1) (eats f7 f8) (eats f8 f2) (eats f8 f7) (craves p0 f7) (craves p0 f8) (craves p1 f7) (craves p2 f1) (craves p3 f4) (craves p3 f8) (craves p4 f7) (craves v0 f4) (craves v0 f5) (craves v1 f3))
(:goal (and (craves p0 f6))))
