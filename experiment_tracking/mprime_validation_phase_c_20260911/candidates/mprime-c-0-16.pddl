(define (problem mprime-c-0-16) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 - pain)
(:init (= (locale f0) 5) (= (locale f1) 3) (= (locale f2) 2) (= (locale f3) 1) (= (locale f4) 2) (= (harmony v0) 2) (= (harmony v1) 3) (eats f0 f1) (eats f0 f3) (eats f0 f4) (eats f1 f2) (eats f1 f3) (eats f1 f4) (eats f2 f0) (eats f2 f1) (eats f2 f4) (eats f3 f0) (eats f3 f2) (eats f3 f4) (eats f4 f0) (eats f4 f1) (eats f4 f2) (craves p0 f3) (craves p1 f3) (craves p1 f4) (craves p2 f2) (craves p3 f0) (craves p3 f2) (craves p4 f1) (craves p5 f0) (craves v0 f3) (craves v0 f4) (craves v1 f0) (craves v1 f4))
(:goal (and (craves p2 f4))))
