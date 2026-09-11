(define (problem mprime-c-0-9) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 - food v0 v1 v2 - pleasure p0 p1 p2 p3 p4 p5 p6 - pain)
(:init (= (locale f0) 3) (= (locale f1) 7) (= (locale f2) 6) (= (locale f3) 2) (= (locale f4) 8) (= (locale f5) 4) (= (locale f6) 2) (= (harmony v0) 3) (= (harmony v1) 2) (= (harmony v2) 3) (eats f0 f1) (eats f0 f2) (eats f1 f6) (eats f2 f3) (eats f3 f0) (eats f3 f2) (eats f3 f4) (eats f3 f5) (eats f3 f6) (eats f4 f5) (eats f5 f1) (eats f5 f3) (eats f6 f0) (eats f6 f1) (eats f6 f2) (craves p0 f2) (craves p1 f0) (craves p1 f3) (craves p2 f0) (craves p2 f6) (craves p3 f4) (craves p4 f4) (craves p5 f2) (craves p5 f4) (craves p6 f1) (craves p6 f5) (craves v0 f6) (craves v1 f2) (craves v2 f5) (craves v2 f6))
(:goal (and (craves p2 f5))))
