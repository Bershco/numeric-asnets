(define (problem mprime-c-0-5) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 v2 v3 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 - pain)
(:init (= (locale f0) 2) (= (locale f1) 2) (= (locale f2) 8) (= (locale f3) 7) (= (locale f4) 6) (= (harmony v0) 1) (= (harmony v1) 3) (= (harmony v2) 2) (= (harmony v3) 3) (eats f0 f3) (eats f0 f4) (eats f1 f0) (eats f1 f2) (eats f2 f0) (eats f2 f4) (eats f3 f0) (eats f3 f1) (eats f4 f0) (eats f4 f2) (eats f4 f3) (craves p0 f1) (craves p1 f1) (craves p1 f4) (craves p2 f3) (craves p3 f3) (craves p3 f4) (craves p4 f1) (craves p5 f1) (craves p5 f4) (craves p6 f2) (craves p7 f4) (craves v0 f4) (craves v1 f2) (craves v2 f2) (craves v2 f3) (craves v3 f0) (craves v3 f2))
(:goal (and (craves p4 f4))))
