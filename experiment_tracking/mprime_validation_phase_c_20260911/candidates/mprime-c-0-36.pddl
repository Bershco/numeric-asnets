(define (problem mprime-c-0-36) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 v2 - pleasure p0 p1 p2 p3 - pain)
(:init (= (locale f0) 9) (= (locale f1) 7) (= (locale f2) 6) (= (locale f3) 0) (= (locale f4) 4) (= (harmony v0) 2) (= (harmony v1) 3) (= (harmony v2) 1) (eats f0 f1) (eats f0 f2) (eats f0 f3) (eats f1 f2) (eats f1 f3) (eats f1 f4) (eats f2 f0) (eats f2 f1) (eats f2 f3) (eats f3 f0) (eats f3 f1) (eats f3 f2) (eats f3 f4) (eats f4 f1) (craves p0 f1) (craves p1 f2) (craves p2 f0) (craves p3 f4) (craves v0 f0) (craves v0 f3) (craves v1 f0) (craves v1 f4) (craves v2 f0) (craves v2 f4))
(:goal (and (craves p0 f4))))
