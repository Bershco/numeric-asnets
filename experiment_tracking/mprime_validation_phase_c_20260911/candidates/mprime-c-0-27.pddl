(define (problem mprime-c-0-27) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 - pain)
(:init (= (locale f0) 1) (= (locale f1) 7) (= (locale f2) 4) (= (locale f3) 2) (= (locale f4) 4) (= (locale f5) 9) (= (locale f6) 1) (= (harmony v0) 2) (= (harmony v1) 1) (eats f0 f2) (eats f1 f0) (eats f1 f4) (eats f2 f5) (eats f3 f0) (eats f4 f3) (eats f4 f5) (eats f5 f0) (eats f5 f3) (eats f5 f4) (eats f5 f6) (eats f6 f0) (eats f6 f1) (eats f6 f3) (craves p0 f0) (craves p1 f2) (craves p2 f2) (craves p3 f4) (craves p4 f4) (craves p4 f6) (craves p5 f3) (craves p6 f0) (craves v0 f3) (craves v1 f3) (craves v1 f6))
(:goal (and (craves p3 f6))))
