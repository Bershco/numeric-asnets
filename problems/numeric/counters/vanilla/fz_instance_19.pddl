(define (problem instance_19)
    (:domain fn-counters)
    (:objects
        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 - counter
    )

    (:init
        (= (max_int) 38)
        (= (value c0) 0)
        (= (value c1) 0)
        (= (value c2) 0)
        (= (value c3) 0)
        (= (value c4) 0)
        (= (value c5) 0)
        (= (value c6) 0)
        (= (value c7) 0)
        (= (value c8) 0)
        (= (value c9) 0)
        (= (value c10) 0)
        (= (value c11) 0)
        (= (value c12) 0)
        (= (value c13) 0)
        (= (value c14) 0)
        (= (value c15) 0)
        (= (value c16) 0)
        (= (value c17) 0)
        (= (value c18) 0)
    )

    (:goal (and
        (<= (+ (value c0) 1) (value c1))
        (<= (+ (value c1) 1) (value c2))
        (<= (+ (value c2) 1) (value c3))
        (<= (+ (value c3) 1) (value c4))
        (<= (+ (value c4) 1) (value c5))
        (<= (+ (value c5) 1) (value c6))
        (<= (+ (value c6) 1) (value c7))
        (<= (+ (value c7) 1) (value c8))
        (<= (+ (value c8) 1) (value c9))
        (<= (+ (value c9) 1) (value c10))
        (<= (+ (value c10) 1) (value c11))
        (<= (+ (value c11) 1) (value c12))
        (<= (+ (value c12) 1) (value c13))
        (<= (+ (value c13) 1) (value c14))
        (<= (+ (value c14) 1) (value c15))
        (<= (+ (value c15) 1) (value c16))
        (<= (+ (value c16) 1) (value c17))
        (<= (+ (value c17) 1) (value c18))
    ))
)
