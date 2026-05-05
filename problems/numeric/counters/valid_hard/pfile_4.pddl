;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_10_100)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 - counter
  )

  (:init
    (= (max_int) 100)
    (= (value c0) 60)
	(= (value c1) 79)
	(= (value c2) 82)
	(= (value c3) 39)
	(= (value c4) 59)
	(= (value c5) 77)
	(= (value c6) 50)
	(= (value c7) 94)
	(= (value c8) 80)
	(= (value c9) 71)
  )

  (:goal (and
    (<= (+ (value c7) 1) (value c8))
	(<= (+ (value c8) 1) (value c3))
	(<= (+ (value c3) 1) (value c1))
	(<= (+ (value c1) 1) (value c5))
	(<= (+ (value c5) 1) (value c4))
	(<= (+ (value c4) 1) (value c2))
	(<= (+ (value c2) 1) (value c9))
	(<= (+ (value c9) 1) (value c6))
	(<= (+ (value c6) 1) (value c0))
  ))


)

