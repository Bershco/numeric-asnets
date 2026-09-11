(define (problem mprime-c-0-2) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 v2 v3 - pleasure p0 p1 p2 p3 p4 p5 p6 - pain)
(:init (= (locale f0) 4) (= (locale f1) 3) (= (locale f2) 6) (= (locale f3) 7) (= (locale f4) 5) (= (harmony v0) 3) (= (harmony v1) 3) (= (harmony v2) 1) (= (harmony v3) 3) (eats f0 f1) (eats f0 f3) (eats f0 f4) (eats f1 f2) (eats f1 f3) (eats f2 f0) (eats f3 f1) (eats f3 f2) (eats f4 f1) (eats f4 f2) (eats f4 f3) (craves p0 f2) (craves p1 f3) (craves p2 f4) (craves p3 f3) (craves p4 f1) (craves p5 f0) (craves p6 f0) (craves v0 f1) (craves v0 f2) (craves v1 f1) (craves v1 f4) (craves v2 f0) (craves v3 f1) (craves v3 f4))
(:goal (and (craves p0 f1))))
