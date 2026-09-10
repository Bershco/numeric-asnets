(define (problem yarin-fo-17)
  (:domain fo-counters)
  (:objects c0 c1 - counter)
  (:init
    (= (max_int) 42)
    (= (value c0) 27)
    (= (value c1) 8)
    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (total-cost) 0))
  (:goal (and
    (<= (+ (value c0) 1) (value c1))))
)
