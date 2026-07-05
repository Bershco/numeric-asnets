(define (problem instance_6)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 - counter
  )

  (:init
    (= (max_int) 10)
    (= (value c0) 4)
    (= (value c1) 3)
    (= (value c2) 5)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c0) 1) (value c1))
    (<= (+ (value c1) 1) (value c2))
  ))
  (:metric minimize (total-cost))
)
