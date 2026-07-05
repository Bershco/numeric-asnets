(define (problem instance_4)
  (:domain fo-counters)
  (:objects
    c0 c1 c2 c3 c4 - counter
  )

  (:init
    (= (max_int) 20)
    (= (value c0) 17)
    (= (value c1) 3)
    (= (value c2) 15)
    (= (value c3) 17)
    (= (value c4) 5)

    (= (rate_value c0) 0)
    (= (rate_value c1) 0)
    (= (rate_value c2) 0)
    (= (rate_value c3) 0)
    (= (rate_value c4) 0)
    (= (total-cost) 0)
  )

  (:goal (and
    (<= (+ (value c0) 1) (value c1))
    (<= (+ (value c1) 1) (value c2))
    (<= (+ (value c2) 1) (value c3))
    (<= (+ (value c3) 1) (value c4))
  ))
  (:metric minimize (total-cost))
)
