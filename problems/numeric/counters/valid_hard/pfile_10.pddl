;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_10_100)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 - counter
  )

  (:init
    (= (max_int) 100)
    (= (value c0) 61)
	(= (value c1) 76)
	(= (value c2) 79)
	(= (value c3) 97)
	(= (value c4) 5)
	(= (value c5) 60)
	(= (value c6) 20)
	(= (value c7) 48)
	(= (value c8) 16)
	(= (value c9) 50)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c6))
	(<= (+ (value c6) 1) (value c2))
	(<= (+ (value c2) 1) (value c7))
	(<= (+ (value c7) 1) (value c9))
	(<= (+ (value c9) 1) (value c0))
	(<= (+ (value c0) 1) (value c5))
	(<= (+ (value c5) 1) (value c3))
	(<= (+ (value c3) 1) (value c8))
	(<= (+ (value c8) 1) (value c4))
  ))


)

