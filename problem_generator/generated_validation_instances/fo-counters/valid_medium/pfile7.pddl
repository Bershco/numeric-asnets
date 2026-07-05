(define (problem instance_9)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 c4 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 14)
    (= (value c1) 9)
    (= (value c2) 13)
    (= (value c3) 1)
    (= (value c4) 13)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (rate_value c4) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c3))
    (<= (+ (value c3) 1) (value c2))
    (<= (+ (value c2) 1) (value c4))
    (<= (+ (value c4) 1) (value c0))
  ))
  (:metric minimize (total-cost))
)
