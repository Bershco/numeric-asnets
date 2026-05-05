;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_4_20)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 4)
	(= (value c1) 13)
	(= (value c2) 3)
	(= (value c3) 4)
  )

  (:goal (and
    (<= (+ (value c2) 1) (value c0))
	(<= (+ (value c0) 1) (value c3))
	(<= (+ (value c3) 1) (value c1))
  ))


)

