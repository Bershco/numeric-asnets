;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_3_20)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 6)
	(= (value c1) 16)
	(= (value c2) 5)
  )

  (:goal (and
    (<= (+ (value c2) 1) (value c1))
	(<= (+ (value c1) 1) (value c0))
  ))


)

