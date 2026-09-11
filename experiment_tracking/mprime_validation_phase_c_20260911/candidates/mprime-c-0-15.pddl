(define (problem mprime-c-0-15) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 - food v0 v1 v2 - pleasure p0 p1 p2 p3 - pain)
(:init (= (locale f0) 3) (= (locale f1) 5) (= (locale f2) 4) (= (locale f3) 2) (= (locale f4) 0) (= (locale f5) 2) (= (harmony v0) 3) (= (harmony v1) 3) (= (harmony v2) 2) (eats f0 f1) (eats f0 f3) (eats f0 f5) (eats f1 f0) (eats f1 f2) (eats f1 f3) (eats f1 f4) (eats f1 f5) (eats f2 f1) (eats f2 f3) (eats f2 f4) (eats f3 f1) (eats f3 f5) (eats f4 f0) (eats f4 f3) (eats f4 f5) (eats f5 f1) (eats f5 f2) (craves p0 f2) (craves p1 f3) (craves p2 f0) (craves p2 f2) (craves p3 f4) (craves v0 f2) (craves v0 f4) (craves v1 f5) (craves v2 f2) (craves v2 f5))
(:goal (and (craves p1 f0))))
