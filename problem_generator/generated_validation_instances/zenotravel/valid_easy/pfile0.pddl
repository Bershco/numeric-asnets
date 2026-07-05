(define (problem ZTRAVEL-1)
(:domain zenotravel)
(:objects
    plane1 - aircraft
    person1 person2 - person
    city0 city1 city2 - city
)
(:init
    (located plane1 city0)
    (= (capacity plane1) 7145)
    (= (fuel plane1) 410)
    (= (slow-burn plane1) 4)
    (= (fast-burn plane1) 16)
    (= (onboard plane1) 0)
    (= (zoom-limit plane1) 6)
    (located person1 city2)
    (located person2 city0)
    (= (distance city0 city0) 0)
    (= (distance city0 city1) 724)
    (= (distance city0 city2) 886)
    (= (distance city1 city0) 724)
    (= (distance city1 city1) 0)
    (= (distance city1 city2) 537)
    (= (distance city2 city0) 886)
    (= (distance city2 city1) 537)
    (= (distance city2 city2) 0)
    (= (total-fuel-used) 0)
)
(:goal (and
    (located person1 city0)
    (located person2 city1)
))
(:metric  minimize (total-fuel-used) )
)
