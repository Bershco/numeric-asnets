(define (problem mprime-c-0-7) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 v2 v3 - pleasure p0 p1 p2 p3 p4 p5 - pain)
(:init (= (locale f0) 2) (= (locale f1) 9) (= (locale f2) 6) (= (locale f3) 3) (= (locale f4) 5) (= (harmony v0) 1) (= (harmony v1) 2) (= (harmony v2) 1) (= (harmony v3) 1) (eats f0 f1) (eats f0 f2) (eats f0 f4) (eats f1 f3) (eats f2 f0) (eats f2 f4) (eats f3 f2) (eats f3 f4) (eats f4 f0) (eats f4 f1) (craves p0 f3) (craves p0 f4) (craves p1 f2) (craves p2 f1) (craves p3 f0) (craves p3 f4) (craves p4 f3) (craves p5 f3) (craves p5 f4) (craves v0 f0) (craves v0 f3) (craves v1 f2) (craves v1 f4) (craves v2 f1) (craves v3 f2) (craves v3 f4))
(:goal (and (craves p0 f0))))
