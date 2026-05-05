;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_6_40)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 - counter
  )

  (:init
    (= (max_int) 40)
    (= (value c0) 4)
	(= (value c1) 36)
	(= (value c2) 12)
	(= (value c3) 3)
	(= (value c4) 25)
	(= (value c5) 7)
  )

  (:goal (and
    (<= (+ (value c2) 1) (value c4))
	(<= (+ (value c4) 1) (value c0))
	(<= (+ (value c0) 1) (value c1))
	(<= (+ (value c1) 1) (value c5))
	(<= (+ (value c5) 1) (value c3))
  ))


)

