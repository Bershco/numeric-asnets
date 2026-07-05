(define (problem instance_3)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 - counter
  )

  (:init
    (= (max_int) 10)
    (= (value c0) 9)
    (= (value c1) 9)
    (= (value c2) 10)
    (= (value c3) 5)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c0) 1) (value c2))
    (<= (+ (value c2) 1) (value c1))
    (<= (+ (value c1) 1) (value c3))
  ))
  (:metric minimize (total-cost))
)
