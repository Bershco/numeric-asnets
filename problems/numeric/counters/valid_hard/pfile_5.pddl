;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_8_100)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 c7 - counter
  )

  (:init
    (= (max_int) 100)
    (= (value c0) 74)
	(= (value c1) 90)
	(= (value c2) 94)
	(= (value c3) 5)
	(= (value c4) 77)
	(= (value c5) 79)
	(= (value c6) 76)
	(= (value c7) 34)
  )

  (:goal (and
    (<= (+ (value c3) 1) (value c7))
	(<= (+ (value c7) 1) (value c4))
	(<= (+ (value c4) 1) (value c6))
	(<= (+ (value c6) 1) (value c0))
	(<= (+ (value c0) 1) (value c1))
	(<= (+ (value c1) 1) (value c5))
	(<= (+ (value c5) 1) (value c2))
  ))


)

