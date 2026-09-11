(define (problem mprime-c-0-32) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 - food v0 v1 v2 - pleasure p0 p1 p2 p3 p4 p5 - pain)
(:init (= (locale f0) 8) (= (locale f1) 5) (= (locale f2) 4) (= (locale f3) 5) (= (locale f4) 6) (= (locale f5) 6) (= (harmony v0) 3) (= (harmony v1) 3) (= (harmony v2) 2) (eats f0 f3) (eats f0 f4) (eats f0 f5) (eats f1 f0) (eats f1 f2) (eats f2 f0) (eats f2 f4) (eats f3 f0) (eats f3 f5) (eats f4 f0) (eats f4 f1) (eats f4 f2) (eats f5 f1) (eats f5 f2) (eats f5 f3) (craves p0 f1) (craves p1 f2) (craves p2 f0) (craves p2 f4) (craves p3 f3) (craves p4 f0) (craves p4 f5) (craves p5 f1) (craves v0 f0) (craves v0 f5) (craves v1 f5) (craves v2 f1))
(:goal (and (craves p0 f0))))
