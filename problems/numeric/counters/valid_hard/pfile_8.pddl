;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_8_100)
  (:domain fn-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 c7 - counter
  )

  (:init
    (= (max_int) 100)
    (= (value c0) 46)
	(= (value c1) 86)
	(= (value c2) 37)
	(= (value c3) 40)
	(= (value c4) 80)
	(= (value c5) 34)
	(= (value c6) 87)
	(= (value c7) 59)
  )

  (:goal (and
    (<= (+ (value c5) 1) (value c4))
	(<= (+ (value c4) 1) (value c3))
	(<= (+ (value c3) 1) (value c6))
	(<= (+ (value c6) 1) (value c2))
	(<= (+ (value c2) 1) (value c0))
	(<= (+ (value c0) 1) (value c1))
	(<= (+ (value c1) 1) (value c7))
  ))


)

