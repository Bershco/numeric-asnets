;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_3_20)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 6)
	(= (value c1) 8)
	(= (value c2) 19)
  )

  (:goal (and
    (<= (+ (value c0) 1) (value c1))
	(<= (+ (value c1) 1) (value c2))
  ))


)

