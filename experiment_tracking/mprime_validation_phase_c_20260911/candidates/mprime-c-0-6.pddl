(define (problem mprime-c-0-6) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 - pain)
(:init (= (locale f0) 1) (= (locale f1) 4) (= (locale f2) 9) (= (locale f3) 4) (= (locale f4) 2) (= (harmony v0) 3) (= (harmony v1) 1) (eats f0 f1) (eats f0 f3) (eats f1 f2) (eats f1 f3) (eats f1 f4) (eats f2 f1) (eats f2 f3) (eats f2 f4) (eats f3 f0) (eats f3 f1) (eats f4 f0) (craves p0 f3) (craves p1 f1) (craves p2 f2) (craves p3 f1) (craves p4 f0) (craves p4 f2) (craves p5 f0) (craves p5 f2) (craves p6 f1) (craves p6 f2) (craves p7 f1) (craves p7 f3) (craves v0 f0) (craves v0 f2) (craves v1 f2) (craves v1 f4))
(:goal (and (craves p4 f3))))
