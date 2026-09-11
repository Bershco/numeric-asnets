(define (problem mprime-c-0-3) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 - pain)
(:init (= (locale f0) 7) (= (locale f1) 8) (= (locale f2) 2) (= (locale f3) 5) (= (locale f4) 7) (= (locale f5) 1) (= (locale f6) 9) (= (harmony v0) 2) (= (harmony v1) 1) (eats f0 f2) (eats f0 f6) (eats f1 f0) (eats f1 f4) (eats f2 f3) (eats f2 f5) (eats f2 f6) (eats f3 f0) (eats f3 f5) (eats f4 f0) (eats f4 f3) (eats f5 f1) (eats f5 f6) (eats f6 f5) (craves p0 f4) (craves p1 f1) (craves p2 f0) (craves p3 f3) (craves p4 f0) (craves p4 f5) (craves p5 f1) (craves p5 f3) (craves p6 f3) (craves p6 f5) (craves p7 f1) (craves v0 f6) (craves v1 f2))
(:goal (and (craves p0 f3))))
