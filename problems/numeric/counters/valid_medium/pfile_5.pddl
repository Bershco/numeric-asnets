;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_5_40)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 - counter
  )

  (:init
    (= (max_int) 40)
    (= (value c0) 36)
	(= (value c1) 10)
	(= (value c2) 8)
	(= (value c3) 40)
	(= (value c4) 0)
  )

  (:goal (and
    (<= (+ (value c2) 1) (value c1))
	(<= (+ (value c1) 1) (value c4))
	(<= (+ (value c4) 1) (value c3))
	(<= (+ (value c3) 1) (value c0))
  ))


)

