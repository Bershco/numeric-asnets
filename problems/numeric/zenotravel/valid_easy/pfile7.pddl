(define (problem ZTRAVEL-8)
(:domain zenotravel)
(:objects
    plane1 - aircraft
    person1 person2 - person
    city0 city1 city2 - city
)
(:init
    (located plane1 city1)
    (= (capacity plane1) 5854)
    (= (fuel plane1) 1998)
    (= (slow-burn plane1) 2)
    (= (fast-burn plane1) 9)
    (= (onboard plane1) 0)
    (= (zoom-limit plane1) 7)
    (located person1 city2)
    (located person2 city0)
    (= (distance city0 city0) 0)
    (= (distance city0 city1) 536)
    (= (distance city0 city2) 768)
    (= (distance city1 city0) 536)
    (= (distance city1 city1) 0)
    (= (distance city1 city2) 548)
    (= (distance city2 city0) 768)
    (= (distance city2 city1) 548)
    (= (distance city2 city2) 0)
    (= (total-fuel-used) 0)
)
(:goal (and
    (located person1 city1)
    (located person2 city2)
))
(:metric  minimize (total-fuel-used) )
)
