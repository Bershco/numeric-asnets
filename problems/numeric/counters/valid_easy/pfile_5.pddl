;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_3_20)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 7)
	(= (value c1) 17)
	(= (value c2) 20)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c0))
	(<= (+ (value c0) 1) (value c2))
  ))


)

