;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_6_40)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 - counter
  )

  (:init
    (= (max_int) 40)
    (= (value c0) 12)
	(= (value c1) 0)
	(= (value c2) 5)
	(= (value c3) 4)
	(= (value c4) 22)
	(= (value c5) 9)
  )

  (:goal (and
    (<= (+ (value c4) 1) (value c0))
	(<= (+ (value c0) 1) (value c2))
	(<= (+ (value c2) 1) (value c3))
	(<= (+ (value c3) 1) (value c1))
	(<= (+ (value c1) 1) (value c5))
  ))


)

