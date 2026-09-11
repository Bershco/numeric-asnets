(define (problem mprime-c-0-39) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 - food v0 v1 v2 v3 - pleasure p0 p1 p2 p3 - pain)
(:init (= (locale f0) 5) (= (locale f1) 0) (= (locale f2) 9) (= (locale f3) 6) (= (locale f4) 3) (= (locale f5) 3) (= (harmony v0) 2) (= (harmony v1) 2) (= (harmony v2) 3) (= (harmony v3) 1) (eats f0 f2) (eats f0 f3) (eats f0 f4) (eats f1 f0) (eats f1 f2) (eats f1 f3) (eats f1 f5) (eats f2 f3) (eats f2 f4) (eats f3 f0) (eats f3 f2) (eats f3 f4) (eats f3 f5) (eats f4 f0) (eats f4 f1) (eats f4 f5) (eats f5 f0) (eats f5 f4) (craves p0 f0) (craves p1 f1) (craves p1 f2) (craves p2 f2) (craves p3 f2) (craves p3 f5) (craves v0 f3) (craves v1 f5) (craves v2 f0) (craves v2 f1) (craves v3 f3))
(:goal (and (craves p2 f3))))
