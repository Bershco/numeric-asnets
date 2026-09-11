(define (problem mprime-c-0-13) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 - food v0 v1 v2 - pleasure p0 p1 p2 p3 p4 p5 p6 - pain)
(:init (= (locale f0) 4) (= (locale f1) 2) (= (locale f2) 2) (= (locale f3) 5) (= (locale f4) 9) (= (locale f5) 1) (= (harmony v0) 2) (= (harmony v1) 2) (= (harmony v2) 2) (eats f0 f2) (eats f0 f3) (eats f1 f0) (eats f1 f4) (eats f1 f5) (eats f2 f0) (eats f2 f3) (eats f2 f4) (eats f3 f0) (eats f3 f5) (eats f4 f0) (eats f4 f1) (eats f4 f3) (eats f5 f3) (eats f5 f4) (craves p0 f4) (craves p0 f5) (craves p1 f0) (craves p1 f4) (craves p2 f2) (craves p2 f3) (craves p3 f4) (craves p4 f1) (craves p4 f2) (craves p5 f0) (craves p6 f0) (craves p6 f4) (craves v0 f1) (craves v0 f3) (craves v1 f0) (craves v2 f0) (craves v2 f2))
(:goal (and (craves p5 f5))))
