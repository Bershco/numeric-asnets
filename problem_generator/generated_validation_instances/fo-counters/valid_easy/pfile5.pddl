(define (problem instance_7)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 - counter
  )

  (:init
    (= (max_int) 10)
    (= (value c0) 2)
    (= (value c1) 5)
    (= (value c2) 3)
    (= (value c3) 10)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c2))
    (<= (+ (value c2) 1) (value c0))
    (<= (+ (value c0) 1) (value c3))
  ))
  (:metric minimize (total-cost))
)
