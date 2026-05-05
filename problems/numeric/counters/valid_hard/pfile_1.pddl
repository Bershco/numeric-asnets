;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_7_100)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 - counter
  )

  (:init
    (= (max_int) 100)
    (= (value c0) 74)
	(= (value c1) 87)
	(= (value c2) 14)
	(= (value c3) 95)
	(= (value c4) 75)
	(= (value c5) 25)
	(= (value c6) 75)
  )

  (:goal (and
    (<= (+ (value c5) 1) (value c3))
	(<= (+ (value c3) 1) (value c6))
	(<= (+ (value c6) 1) (value c4))
	(<= (+ (value c4) 1) (value c2))
	(<= (+ (value c2) 1) (value c0))
	(<= (+ (value c0) 1) (value c1))
  ))


)

