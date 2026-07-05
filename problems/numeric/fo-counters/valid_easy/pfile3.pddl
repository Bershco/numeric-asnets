(define (problem instance_5)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 - counter
  )

  (:init
    (= (max_int) 10)
    (= (value c0) 8)
    (= (value c1) 6)
    (= (value c2) 9)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c2) 1) (value c0))
    (<= (+ (value c0) 1) (value c1))
  ))
  (:metric minimize (total-cost))
)
