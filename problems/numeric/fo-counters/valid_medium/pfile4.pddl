(define (problem instance_6)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 c4 c5 c6 c7 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 10)
    (= (value c1) 0)
    (= (value c2) 10)
    (= (value c3) 18)
    (= (value c4) 13)
    (= (value c5) 16)
    (= (value c6) 16)
    (= (value c7) 17)

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
    (<= (+ (value c6) 1) (value c3))
    (<= (+ (value c3) 1) (value c2))
    (<= (+ (value c2) 1) (value c1))
    (<= (+ (value c1) 1) (value c0))
    (<= (+ (value c0) 1) (value c4))
    (<= (+ (value c4) 1) (value c5))
    (<= (+ (value c5) 1) (value c7))
  ))
  (:metric minimize (total-cost))
)
