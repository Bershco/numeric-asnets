(define (problem instance_8)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 c7 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 20)
    (= (value c1) 1)
    (= (value c2) 12)
    (= (value c3) 6)
    (= (value c4) 12)
    (= (value c5) 10)
    (= (value c6) 12)
    (= (value c7) 16)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (rate_value c4) 0)
    (= (rate_value c5) 0)
    (= (rate_value c6) 0)
    (= (rate_value c7) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c1) 1) (value c7))
    (<= (+ (value c7) 1) (value c2))
    (<= (+ (value c2) 1) (value c3))
    (<= (+ (value c3) 1) (value c4))
    (<= (+ (value c4) 1) (value c0))
    (<= (+ (value c0) 1) (value c6))
    (<= (+ (value c6) 1) (value c5))
  ))
  (:metric minimize (total-cost))
)
