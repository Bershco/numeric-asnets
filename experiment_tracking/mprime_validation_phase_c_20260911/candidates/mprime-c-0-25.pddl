(define (problem mprime-c-0-25) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 v2 v3 - pleasure p0 p1 p2 p3 p4 - pain)
(:init (= (locale f0) 5) (= (locale f1) 5) (= (locale f2) 3) (= (locale f3) 8) (= (locale f4) 2) (= (harmony v0) 2) (= (harmony v1) 2) (= (harmony v2) 3) (= (harmony v3) 1) (eats f0 f2) (eats f0 f3) (eats f1 f0) (eats f1 f2) (eats f1 f4) (eats f2 f0) (eats f2 f1) (eats f2 f3) (eats f2 f4) (eats f3 f1) (eats f3 f4) (eats f4 f1) (eats f4 f3) (craves p0 f0) (craves p1 f3) (craves p2 f3) (craves p3 f2) (craves p4 f4) (craves v0 f0) (craves v1 f3) (craves v2 f4) (craves v3 f0))
(:goal (and (craves p4 f2))))
