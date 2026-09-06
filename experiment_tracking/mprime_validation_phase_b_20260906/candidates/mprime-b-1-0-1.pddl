(define (problem mprime-b-1-0-1) (:domain mystery-prime-typed)
(:objects f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 - food v0 v1 - pleasure p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 - pain)
(:init (= (locale f0) 5) (= (locale f1) 5) (= (locale f2) 4) (= (locale f3) 1) (= (locale f4) 3) (= (locale f5) 1) (= (locale f6) 1) (= (locale f7) 3) (= (locale f8) 1) (= (locale f9) 1) (= (harmony v0) 1) (= (harmony v1) 3) (eats f0 f3) (eats f1 f4) (eats f2 f1) (eats f3 f7) (eats f3 f8) (eats f4 f8) (eats f5 f2) (eats f6 f9) (eats f7 f5) (eats f8 f6) (eats f9 f0) (eats f9 f6) (craves p0 f2) (craves p1 f2) (craves p2 f3) (craves p3 f4) (craves p4 f6) (craves p5 f6) (craves p6 f8) (craves p7 f7) (craves p8 f0) (craves p9 f1) (craves v0 f8) (craves v1 f4))
(:goal (and (craves p7 f3))))
