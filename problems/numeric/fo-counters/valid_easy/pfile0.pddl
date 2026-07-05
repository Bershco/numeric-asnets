(define (problem instance_2)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 - counter
  )

  (:init
    (= (max_int) 10)
    (= (value c0) 6)
    (= (value c1) 0)
    (= (value c2) 8)
    (= (value c3) 3)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c0))
    (<= (+ (value c0) 1) (value c3))
    (<= (+ (value c3) 1) (value c2))
  ))
  (:metric minimize (total-cost))
)
