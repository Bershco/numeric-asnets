(define (problem instance_2)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 c4 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 13)
    (= (value c1) 0)
    (= (value c2) 19)
    (= (value c3) 18)
    (= (value c4) 17)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (rate_value c4) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c0))
    (<= (+ (value c0) 1) (value c4))
    (<= (+ (value c4) 1) (value c2))
    (<= (+ (value c2) 1) (value c3))
  ))
  (:metric minimize (total-cost))
)
