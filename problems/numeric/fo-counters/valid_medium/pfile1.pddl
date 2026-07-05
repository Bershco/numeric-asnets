(define (problem instance_3)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 6)
    (= (value c1) 10)
    (= (value c2) 14)
    (= (value c3) 14)
    (= (value c4) 17)
    (= (value c5) 7)
    (= (value c6) 17)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (rate_value c4) 0)
    (= (rate_value c5) 0)
    (= (rate_value c6) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c6) 1) (value c1))
    (<= (+ (value c1) 1) (value c0))
    (<= (+ (value c0) 1) (value c2))
    (<= (+ (value c2) 1) (value c3))
    (<= (+ (value c3) 1) (value c5))
    (<= (+ (value c5) 1) (value c4))
  ))
  (:metric minimize (total-cost))
)
