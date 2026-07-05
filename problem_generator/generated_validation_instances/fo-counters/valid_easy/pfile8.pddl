(define (problem instance_10)
  (:domain fo-counters)
  (:objects
    c0 c1 - counter
  )

  (:init
    (= (max_int) 10)
    (= (value c0) 2)
    (= (value c1) 5)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c0))
  ))
  (:metric minimize (total-cost))
)
