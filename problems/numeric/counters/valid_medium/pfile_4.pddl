;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_5_40)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 - counter
  )

  (:init
    (= (max_int) 40)
    (= (value c0) 3)
	(= (value c1) 12)
	(= (value c2) 21)
	(= (value c3) 37)
	(= (value c4) 23)
  )

  (:goal (and
    (<= (+ (value c2) 1) (value c1))
	(<= (+ (value c1) 1) (value c0))
	(<= (+ (value c0) 1) (value c3))
	(<= (+ (value c3) 1) (value c4))
  ))


)

